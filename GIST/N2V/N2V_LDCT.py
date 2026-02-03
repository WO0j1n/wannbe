# ============================================================
# Noise2Void (N2V) - steps 기반 학습 (PyTorch)
# LDCT 버전:
#   - Train: Quarter Dose(QD) noisy-only
#   - Test : QD -> denoise, compare with Full Dose(FD) as GT
#   - Save:
#       (1) full-view overlay (QD vs FD with crop rectangle)
#       (2) crop panel (QD crop | denoised crop | FD crop)
# ============================================================

import os
import re
import math
import random
from glob import glob
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False

# ---------------------------
# 0) Device (cuda / mps / cpu)
# ---------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("🔥 device:", device)


# ---------------------------
# 1) Image I/O
# ---------------------------
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(root: str) -> List[str]:
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(files)


def is_valid_png(path: str) -> bool:
    """
    PNG가 정상인지 빠르게 검사:
    - Image.open(...).verify(): 파일 구조 검사(디코드까지는 아님)
    - 이후 다시 열어서 load(): 실제 디코딩까지 확인
    """
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False

def filter_valid_images(paths: List[str]) -> List[str]:
    ok = []
    bad = []
    for p in tqdm(paths, desc="Scan PNG validity", ncols=110):
        if is_valid_png(p):
            ok.append(p)
        else:
            bad.append(p)

    print(f"✅ valid: {len(ok)} | ❌ broken: {len(bad)}")
    if len(bad) > 0:
        print("---- broken samples ----")
        for b in bad[:10]:
            print("BAD:", b)
    return ok

def load_image(path: str, grayscale: bool = True) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert("L") if grayscale else img.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    if arr.ndim == 2:
        arr = arr[None, ...]  # [1,H,W]
    else:
        arr = arr.transpose(2, 0, 1)  # [C,H,W]
    return torch.from_numpy(arr)


def to_u8(x01: np.ndarray) -> np.ndarray:
    return (np.clip(x01, 0, 1) * 255.0).astype(np.uint8)


# ---------------------------
# 2) Pair matching (FD <-> QD)
# ---------------------------
def make_pair_key(path: str) -> str:
    base = os.path.basename(path)

    # 1) L### 같은 케이스 ID 추출 (없으면 UNKNOWN)
    m_case = re.search(r"(L\d+)", base)
    case_id = m_case.group(1) if m_case else "UNKNOWN"

    # 2) CT.####.#### 패턴에서 두 번째 ####(slice id) 추출
    #    예) ...CT.0002.0001.... / ...CT.0004.0001....
    m_ct = re.search(r"\.CT\.(\d{4})\.(\d{4})\.", base)
    if m_ct:
        slice_id = m_ct.group(2)   # ✅ 두 번째가 slice index
        return f"{case_id}_slice{slice_id}"

    # 3) 혹시 CT 패턴이 다르면 fallback: 파일명에서 숫자 4자리 중 마지막 후보 하나
    #    (데이터가 다른 형식이면 여기 조금 더 커스터마이징 필요)
    m_last4 = re.findall(r"(\d{4})", base)
    if len(m_last4) > 0:
        return f"{case_id}_slice{m_last4[-1]}"

    # 4) 최후 fallback
    return f"{case_id}_{base}"


def build_fd_qd_pairs(fd_root: str, qd_root: str) -> List[Tuple[str, str]]:
    """
    return: [(qd_path, fd_path), ...]
    """
    # 0) 먼저 파일 리스트 만들기
    fd_paths = list_images(fd_root)
    qd_paths = list_images(qd_root)

    # 1) 그 다음에 깨진 파일 제거(스캔)
    fd_paths = filter_valid_images(fd_paths)
    qd_paths = filter_valid_images(qd_paths)

    # 2) 기본 체크
    if len(fd_paths) == 0:
        raise RuntimeError(f"No FD images found: {fd_root}")
    if len(qd_paths) == 0:
        raise RuntimeError(f"No QD images found: {qd_root}")

    # 3) FD를 key로 맵핑
    fd_map: Dict[str, str] = {}
    dup = 0
    for p in fd_paths:
        k = make_pair_key(p)
        if k in fd_map:
            dup += 1
        fd_map[k] = p
    if dup > 0:
        print(f"[WARN] FD duplicate keys overwritten: {dup}")

    # 4) QD를 돌면서 매칭
    pairs = []
    miss = 0
    for q in qd_paths:
        k = make_pair_key(q)
        if k in fd_map:
            pairs.append((q, fd_map[k]))
        else:
            miss += 1

    # 5) 디버그 + 예외
    if len(pairs) == 0:
        print("\n[DEBUG] FD key samples:")
        for p in fd_paths[:5]:
            print(" FD:", os.path.basename(p))
            print("  ->", make_pair_key(p))

        print("\n[DEBUG] QD key samples:")
        for p in qd_paths[:5]:
            print(" QD:", os.path.basename(p))
            print("  ->", make_pair_key(p))

        raise RuntimeError(
            "No matched QD-FD pairs found.\n"
            f"fd_root={fd_root}\nqd_root={qd_root}\n"
            "Check naming rules in make_pair_key()."
        )

    print(f"✅ pairs: {len(pairs)} (QD matched), miss_qd={miss}")
    return pairs



# ---------------------------
# 3) Stratified sampling for N points
# ---------------------------
def stratified_sample_coords(patch_h: int, patch_w: int, n_points: int) -> List[Tuple[int, int]]:
    g = int(math.sqrt(n_points))
    g = max(g, 1)
    if g * g < n_points:
        g += 1

    cell_h = patch_h / g
    cell_w = patch_w / g

    coords = []
    for gy in range(g):
        for gx in range(g):
            y0 = int(round(gy * cell_h))
            y1 = int(round((gy + 1) * cell_h))
            x0 = int(round(gx * cell_w))
            x1 = int(round((gx + 1) * cell_w))

            y1 = min(y1, patch_h)
            x1 = min(x1, patch_w)
            if y0 >= y1 or x0 >= x1:
                continue

            yy = random.randint(y0, y1 - 1)
            xx = random.randint(x0, x1 - 1)
            coords.append((yy, xx))

    random.shuffle(coords)
    if len(coords) >= n_points:
        coords = coords[:n_points]
    else:
        need = n_points - len(coords)
        for _ in range(need):
            coords.append((random.randint(0, patch_h - 1), random.randint(0, patch_w - 1)))
    return coords


# ---------------------------
# 4) N2V masking (blind-spot replacement)
# ---------------------------
def apply_n2v_masking(
    patch: torch.Tensor,
    coords: List[Tuple[int, int]],
    radius: int = 7,   # ✅ neighbor_radius >= 7
) -> Tuple[torch.Tensor, torch.Tensor]:
    C, H, W = patch.shape
    masked = patch.clone()
    mask = torch.zeros((1, H, W), dtype=torch.float32)

    for (y, x) in coords:
        mask[0, y, x] = 1.0

        # 주변 값으로 치환 (최대 10번 시도)
        for _ in range(10):
            dy = random.randint(-radius, radius)
            dx = random.randint(-radius, radius)
            ny = y + dy
            nx = x + dx
            if 0 <= ny < H and 0 <= nx < W and (dy != 0 or dx != 0):
                masked[:, y, x] = patch[:, ny, nx]
                break
        else:
            # 실패 시 랜덤 fallback
            yy = random.randint(0, H - 1)
            xx = random.randint(0, W - 1)
            if yy == y and xx == x:
                yy = (yy + 1) % H
            masked[:, y, x] = patch[:, yy, xx]

    return masked, mask


# ---------------------------
# 5) Dataset: random patch from QD only (noisy-only)
# ---------------------------
class LDCTN2VPatchDataset(Dataset):
    def __init__(self, qd_paths, patch_size=240, n_masked=256,
                 grayscale=True, neighbor_radius=7, augment=True, max_retry=20):
        self.qd_paths = list(qd_paths)
        self.patch_size = patch_size
        self.n_masked = n_masked
        self.grayscale = grayscale
        self.neighbor_radius = neighbor_radius
        self.augment = augment
        self.max_retry = max_retry

    def __len__(self):
        return len(self.qd_paths)

    def _random_crop(self, img: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
        C, H, W = img.shape
        ps = self.patch_size

        if H < ps or W < ps:
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
            C, H, W = img.shape

        y0 = random.randint(0, H - ps)
        x0 = random.randint(0, W - ps)
        patch = img[:, y0:y0 + ps, x0:x0 + ps]
        return patch, (x0, y0, x0+ps, y0+ps)

    def _augment(self, patch: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return patch
        if random.random() < 0.5:
            patch = torch.flip(patch, dims=[2])
        if random.random() < 0.5:
            patch = torch.flip(patch, dims=[1])
        k = random.randint(0, 3)
        patch = torch.rot90(patch, k=k, dims=[1, 2])
        return patch

    def __getitem__(self, idx):
        # ✅ 어떤 파일이 깨져도 학습이 멈추지 않게 retry
        for _ in range(self.max_retry):
            path = random.choice(self.qd_paths)
            try:
                img = load_image(path, grayscale=self.grayscale)
            except Exception:
                continue  # 깨진 파일이면 다른 파일로

            try:
                patch, _ = self._random_crop(img)
                patch = self._augment(patch)

                coords = stratified_sample_coords(self.patch_size, self.patch_size, self.n_masked)
                masked_patch, mask = apply_n2v_masking(patch, coords, radius=self.neighbor_radius)

                target = patch
                return masked_patch, target, mask
            except Exception:
                continue

        # 여기까지 왔으면 "너무 많은 파일이 깨졌거나" 뭔가 구조가 이상한 상태
        raise RuntimeError("Too many failed reads/crops. Dataset may contain many corrupted images.")


def split_pairs(pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]

    return train_pairs, val_pairs, test_pairs



# ---------------------------
# 6) U-Net
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.relu(self.bn(self.conv(x)))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, out_channels: int = 1, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.enc0_1 = ConvBlock(in_channels, base_channels, kernel_size, padding)
        self.enc0_2 = ConvBlock(base_channels, base_channels, kernel_size, padding)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc1_1 = ConvBlock(base_channels, base_channels * 2, kernel_size, padding)
        self.enc1_2 = ConvBlock(base_channels * 2, base_channels * 2, kernel_size, padding)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.mid1 = ConvBlock(base_channels * 2, base_channels * 4, kernel_size, padding)
        self.mid2 = ConvBlock(base_channels * 4, base_channels * 4, kernel_size, padding)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1_1 = ConvBlock(base_channels * 4, base_channels * 2, kernel_size, padding)
        self.dec1_2 = ConvBlock(base_channels * 2, base_channels * 2, kernel_size, padding)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec0_1 = ConvBlock(base_channels * 2, base_channels, kernel_size, padding)
        self.dec0_2 = ConvBlock(base_channels, base_channels, kernel_size, padding)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        enc0 = self.enc0_2(self.enc0_1(x))

        enc1 = self.pool1(enc0)
        enc1 = self.enc1_2(self.enc1_1(enc1))

        mid = self.pool2(enc1)
        mid = self.mid2(self.mid1(mid))

        dec1 = self.upconv1(mid)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1_2(self.dec1_1(dec1))

        dec0 = self.upconv2(dec1)
        dec0 = torch.cat([dec0, enc0], dim=1)
        dec0 = self.dec0_2(self.dec0_1(dec0))

        return self.out_conv(dec0)


# ---------------------------
# 7) Masked MSE loss
# ---------------------------
def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.repeat(1, pred.shape[1], 1, 1)

    mse_map = (pred - target) ** 2
    masked = mse_map * mask
    return masked.sum() / (mask.sum() + eps)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


# ---------------------------
# 8) Train (steps-based)
# ---------------------------
def train_n2v_steps_ldct(
    qd_train_paths: List[str],
    grayscale: bool = True,
    patch_size: int = 240,
    n_masked: int = 256,
    neighbor_radius: int = 7,
    base_channels: int = 32,
    batch_size: int = 16,
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    steps: int = 20000,
    log_every: int = 100,
    ckpt_path: str = "./n2v_ldct_ckpt.pth",
    seed: int = 0,
    pbar_mode: str = "loss",   # "loss" | "avg_loss" | "lr"
    lr_in_desc: bool = True,
) -> nn.Module:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    in_ch = 1 if grayscale else 3

    dataset = LDCTN2VPatchDataset(
        qd_paths=qd_train_paths,
        patch_size=patch_size,
        n_masked=n_masked,
        grayscale=grayscale,
        neighbor_radius=neighbor_radius,
        augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    if len(loader) == 0:
        raise RuntimeError("DataLoader has 0 batches. Reduce batch_size or check dataset.")

    model = UNet(in_channels=in_ch, base_channels=base_channels, out_channels=in_ch, kernel_size=3).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )

    model.train()
    it = iter(loader)
    running = 0.0

    base_desc = "Train(N2V-LDCT)"
    pbar = tqdm(range(1, steps + 1), desc=base_desc, ncols=120)

    for step in pbar:
        try:
            x_in, x_tgt, m = next(it)
        except StopIteration:
            it = iter(loader)
            x_in, x_tgt, m = next(it)

        x_in = x_in.to(device, non_blocking=True)
        x_tgt = x_tgt.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        pred = model(x_in)
        loss = masked_mse_loss(pred, x_tgt, m)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        cur_lr = get_lr(optimizer)

        if pbar_mode == "loss":
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        elif pbar_mode == "avg_loss":
            if step % log_every == 0:
                avg = running / log_every
                running = 0.0
                pbar.set_postfix(avg_loss=f"{avg:.6f}")

        elif pbar_mode == "lr":
            pbar.set_postfix(lr=f"{cur_lr:.2e}")
            if lr_in_desc:
                pbar.set_description(f"{base_desc} | loss={loss.item():.4f}")
            else:
                pbar.set_description(base_desc)

        else:
            raise ValueError('pbar_mode must be one of: "loss", "avg_loss", "lr"')

        if pbar_mode != "avg_loss" and (step % log_every == 0):
            running = 0.0

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": steps,
            "grayscale": grayscale,
            "in_ch": in_ch,
            "base_channels": base_channels,
            "patch_size": patch_size,
            "n_masked": n_masked,
            "neighbor_radius": neighbor_radius,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        ckpt_path,
    )
    print(f"\n✅ Saved checkpoint -> {ckpt_path}")
    return model


# ---------------------------
# 9) Load checkpoint
# ---------------------------
def load_n2v_checkpoint(ckpt_path: str, device_override: Optional[torch.device] = None) -> nn.Module:
    dev = device_override if device_override is not None else device
    ckpt = torch.load(ckpt_path, map_location=dev)

    in_ch = ckpt.get("in_ch", 1)
    base_channels = ckpt.get("base_channels", 32)

    model = UNet(in_channels=in_ch, base_channels=base_channels, out_channels=in_ch, kernel_size=3).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint -> {ckpt_path} (device={dev})")
    return model


# ---------------------------
# 10) PSNR (float/uint8)
# ---------------------------
def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0, eps: float = 1e-12) -> float:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse < eps:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


@torch.no_grad()
def denoise_tensor_full(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x: [1,C,H,W] in [0,1]
    return: [1,C,H,W] in [0,1]
    """
    model.eval()
    y = model(x).clamp(0, 1)
    return y


# ---------------------------
# 11) Visualization helpers (crop 위치 표시 + 패널 저장)
# ---------------------------
def draw_rect_on_u8(img_u8: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """
    img_u8: [H,W] grayscale uint8
    rect: (x0,y0,x1,y1)
    """
    pil = Image.fromarray(img_u8, mode="L").convert("RGB")
    dr = ImageDraw.Draw(pil)
    x0, y0, x1, y1 = rect
    dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
    return np.array(pil, dtype=np.uint8)


def make_triptych(a_u8: np.ndarray, b_u8: np.ndarray, c_u8: np.ndarray, title: str = "") -> Image.Image:
    """
    a,b,c: [H,W] grayscale uint8
    return: PIL RGB
    """
    A = Image.fromarray(a_u8, mode="L").convert("RGB")
    B = Image.fromarray(b_u8, mode="L").convert("RGB")
    C = Image.fromarray(c_u8, mode="L").convert("RGB")

    w, h = A.size
    canvas = Image.new("RGB", (w * 3, h), (0, 0, 0))
    canvas.paste(A, (0, 0))
    canvas.paste(B, (w, 0))
    canvas.paste(C, (w * 2, 0))

    if title:
        dr = ImageDraw.Draw(canvas)
        dr.text((8, 8), title, fill=(255, 255, 0))
    return canvas


def make_side_by_side(a_rgb_u8: np.ndarray, b_rgb_u8: np.ndarray, title: str = "") -> Image.Image:
    A = Image.fromarray(a_rgb_u8, mode="RGB")
    B = Image.fromarray(b_rgb_u8, mode="RGB")
    w, h = A.size
    canvas = Image.new("RGB", (w * 2, h), (0, 0, 0))
    canvas.paste(A, (0, 0))
    canvas.paste(B, (w, 0))
    if title:
        dr = ImageDraw.Draw(canvas)
        dr.text((8, 8), title, fill=(255, 255, 0))
    return canvas


# ---------------------------
# 12) Test: QD -> denoise, compare with FD
#     + 저장: (full overlay) + (crop panel)
# ---------------------------
@torch.no_grad()
def run_test_ldct_pairs(
    model: nn.Module,
    pairs: List[Tuple[str, str]],   # [(qd_path, fd_path), ...]
    out_dir: str = "./ldct_test_outputs",
    grayscale: bool = True,
    crop_size: int = 240,
    save_full_overlay: bool = True,
    save_crop_panel: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ov_dir = os.path.join(out_dir, "overlay_full")
    cp_dir = os.path.join(out_dir, "crop_panels")
    dn_dir = os.path.join(out_dir, "denoised_full")
    os.makedirs(ov_dir, exist_ok=True)
    os.makedirs(cp_dir, exist_ok=True)
    os.makedirs(dn_dir, exist_ok=True)

    psnr_full_list = []
    psnr_crop_list = []

    for (qd_path, fd_path) in tqdm(pairs, desc="Test(LDCT QD->FD)", ncols=120):
        # load
        qd = load_image(qd_path, grayscale=grayscale)  # [1,H,W]
        fd = load_image(fd_path, grayscale=grayscale)

        # ensure same size (혹시 다르면 FD 기준으로 맞춤)
        C, H, W = qd.shape
        _, H2, W2 = fd.shape
        Hm = min(H, H2)
        Wm = min(W, W2)
        qd = qd[:, :Hm, :Wm]
        fd = fd[:, :Hm, :Wm]
        H, W = Hm, Wm

        # denoise full
        x = qd.unsqueeze(0).to(next(model.parameters()).device)  # [1,1,H,W]
        y = denoise_tensor_full(model, x)[0].detach().cpu().numpy()  # [1,H,W]

        qd_np = qd.numpy()
        fd_np = fd.numpy()

        # PSNR full (in [0,1])
        psnr_full_list.append(psnr(y, fd_np, data_range=1.0))

        # save denoised full
        y_u8 = to_u8(y[0])
        fname = os.path.basename(qd_path)
        Image.fromarray(y_u8, mode="L").save(os.path.join(dn_dir, fname))

        # choose a crop region (random) for "where is this patch?"
        cs = crop_size
        if H < cs or W < cs:
            # 작으면 그냥 전체를 crop 취급
            x0, y0 = 0, 0
            x1, y1 = W, H
        else:
            x0 = random.randint(0, W - cs)
            y0 = random.randint(0, H - cs)
            x1 = x0 + cs
            y1 = y0 + cs

        # crop for panel
        qd_crop = qd_np[0, y0:y1, x0:x1]
        fd_crop = fd_np[0, y0:y1, x0:x1]
        y_crop = y[0, y0:y1, x0:x1]
        psnr_crop_list.append(psnr(y_crop, fd_crop, data_range=1.0))

        qd_crop_u8 = to_u8(qd_crop)
        fd_crop_u8 = to_u8(fd_crop)
        y_crop_u8 = to_u8(y_crop)

        # (1) full overlay: QD vs FD with rectangle
        if save_full_overlay:
            rect = (x0, y0, x1, y1)
            qd_full_u8 = to_u8(qd_np[0])
            fd_full_u8 = to_u8(fd_np[0])
            qd_mark = draw_rect_on_u8(qd_full_u8, rect)
            fd_mark = draw_rect_on_u8(fd_full_u8, rect)

            title = f"LEFT:QD | RIGHT:FD  crop=({x0},{y0})-({x1},{y1})"
            canvas = make_side_by_side(qd_mark, fd_mark, title=title)
            canvas.save(os.path.join(ov_dir, fname.replace(".png", f"_xy{x0}_{y0}.png")))

        # (2) crop panel: QD crop | denoised crop | FD crop
        if save_crop_panel:
            title = f"QD | DENOISED | FD    PSNR(crop)={psnr_crop_list[-1]:.2f}dB"
            panel = make_triptych(qd_crop_u8, y_crop_u8, fd_crop_u8, title=title)
            panel.save(os.path.join(cp_dir, fname.replace(".png", f"_crop_xy{x0}_{y0}.png")))

    print("\n====================")
    print("TEST DONE (LDCT)")
    print(f"Saved denoised full : {dn_dir}")
    print(f"Saved overlays      : {ov_dir}")
    print(f"Saved crop panels   : {cp_dir}")

    if len(psnr_full_list) > 0:
        print(f"PSNR FULL mean: {float(np.mean(psnr_full_list)):.4f} dB")
        print(f"PSNR FULL std : {float(np.std(psnr_full_list)):.4f} dB")
        print(f"PSNR FULL min/max: {float(np.min(psnr_full_list)):.4f} / {float(np.max(psnr_full_list)):.4f} dB")

    if len(psnr_crop_list) > 0:
        print(f"PSNR CROP mean: {float(np.mean(psnr_crop_list)):.4f} dB")
        print(f"PSNR CROP std : {float(np.std(psnr_crop_list)):.4f} dB")
        print(f"PSNR CROP min/max: {float(np.min(psnr_crop_list)):.4f} / {float(np.max(psnr_crop_list)):.4f} dB")
    print("====================\n")


if __name__ == "__main__":

    FD_DIR = "/home/work/LDCT/Sharp Kernel (D45)/L067"
    QD_DIR = "/home/work/LDCT/Sharp Kernel (D45) quater/L067"

    # =========================
    # ✅ 실험 설정 먼저 정의 (seed 포함!)
    # =========================
    grayscale = True
    patch_size = 240
    neighbor_radius = 7
    n_masked = 256
    base_channels = 32
    batch_size = 16
    lr = 2e-4
    weight_decay = 1e-5
    steps = 20000
    log_every = 100
    seed = 0  # ✅ 여기로 올려야 NameError 안 남

    ckpt_path = "./n2v_ldct_qd_only.pth"
    out_dir = "./ldct_outputs_qd2fd"

    PBAR_MODE = "loss"
    LR_IN_DESC = True

    DO_TRAIN = True
    DO_TEST = True

    # =========================
    # ✅ Pair list 만들기
    # =========================
    pairs = build_fd_qd_pairs(FD_DIR, QD_DIR)

    # ✅ train/val/test split
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=seed
    )

    # ✅ train에는 train_pairs의 QD만
    qd_train_paths = [q for (q, f) in train_pairs]

    print(f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # =========================
    # ✅ TRAIN
    # =========================
    if DO_TRAIN:
        print("\n====================")
        print("TRAIN: N2V on LDCT (QD noisy-only)")
        print("====================")
        print("QD_DIR   :", QD_DIR)
        print("FD_DIR(GT):", FD_DIR)
        print("pairs(all):", len(pairs))
        print("train_qd :", len(qd_train_paths))
        print("patch    :", patch_size)
        print("radius   :", neighbor_radius)
        print("batch    :", batch_size)
        print("steps    :", steps)
        print("====================\n")

        model = train_n2v_steps_ldct(
            qd_train_paths=qd_train_paths,
            grayscale=grayscale,
            patch_size=patch_size,
            n_masked=n_masked,
            neighbor_radius=neighbor_radius,
            base_channels=base_channels,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            steps=steps,
            log_every=log_every,
            ckpt_path=ckpt_path,
            seed=seed,
            pbar_mode=PBAR_MODE,
            lr_in_desc=LR_IN_DESC,
        )
    else:
        model = load_n2v_checkpoint(ckpt_path)

    # =========================
    # ✅ TEST (test_pairs만!)
    # =========================
    if DO_TEST:
        print("\n====================")
        print("TEST: QD -> denoise, compare with FD (GT)")
        print("====================")
        print("out_dir  :", out_dir)
        print("test_pairs:", len(test_pairs))
        print("crop_size:", patch_size, "(used for visualization crop)")
        print("====================\n")

        model = load_n2v_checkpoint(ckpt_path)

        run_test_ldct_pairs(
            model=model,
            pairs=test_pairs,
            out_dir=out_dir,
            grayscale=grayscale,
            crop_size=patch_size,
            save_full_overlay=True,
            save_crop_panel=True,
        )

