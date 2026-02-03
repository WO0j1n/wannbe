# ============================================================
# Noise2Void (N2V) - "논문 스타일" steps 기반 학습 버전 (PyTorch)
# + tqdm progress bar modes:
#   1) loss only
#   2) avg_loss only
#   3) lr included (lr only OR lr+metric in desc)
# ============================================================

import os
import math
import random
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


# ---------------------------
# 0) Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# 1) Image I/O
# ---------------------------
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(root: str) -> List[str]:
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(files)


def load_image(path: str, grayscale: bool = True) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert("L") if grayscale else img.convert("RGB")

    arr = np.array(img, dtype=np.float32) / 255.0

    if arr.ndim == 2:
        arr = arr[None, ...]  # [1,H,W]
    else:
        arr = arr.transpose(2, 0, 1)  # [C,H,W]

    return torch.from_numpy(arr)


# ---------------------------
# 2) Stratified sampling for N points
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
# 3) N2V masking (blind-spot replacement)
# ---------------------------
def apply_n2v_masking(
    patch: torch.Tensor,
    coords: List[Tuple[int, int]],
    radius: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    C, H, W = patch.shape
    masked = patch.clone()
    mask = torch.zeros((1, H, W), dtype=torch.float32)

    for (y, x) in coords:
        mask[0, y, x] = 1.0

        for _ in range(10):
            dy = random.randint(-radius, radius)
            dx = random.randint(-radius, radius)
            ny = y + dy
            nx = x + dx
            if 0 <= ny < H and 0 <= nx < W and (dy != 0 or dx != 0):
                masked[:, y, x] = patch[:, ny, nx]
                break
        else:
            yy = random.randint(0, H - 1)
            xx = random.randint(0, W - 1)
            if yy == y and xx == x:
                yy = (yy + 1) % H
            masked[:, y, x] = patch[:, yy, xx]

    return masked, mask


# ---------------------------
# 4) Dataset: random patch + masking
# ---------------------------
class N2VPatchDataset(Dataset):
    def __init__(
        self,
        image_root_or_list,
        patch_size: int = 64,
        n_masked: int = 64,
        grayscale: bool = True,
        neighbor_radius: int = 5,
        augment: bool = True,
    ):
        if isinstance(image_root_or_list, str):
            image_paths = list_images(image_root_or_list)
        else:
            image_paths = list(image_root_or_list)

        if len(image_paths) == 0:
            raise RuntimeError("No images found. Check your dataset path/root.")

        self.image_paths = image_paths
        self.patch_size = patch_size
        self.n_masked = n_masked
        self.grayscale = grayscale
        self.neighbor_radius = neighbor_radius
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def _random_crop(self, img: torch.Tensor) -> torch.Tensor:
        C, H, W = img.shape
        ps = self.patch_size

        if H < ps or W < ps:
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
            C, H, W = img.shape

        y0 = random.randint(0, H - ps)
        x0 = random.randint(0, W - ps)
        return img[:, y0:y0 + ps, x0:x0 + ps]

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
        path = random.choice(self.image_paths)
        img = load_image(path, grayscale=self.grayscale)

        patch = self._random_crop(img)
        patch = self._augment(patch)

        coords = stratified_sample_coords(self.patch_size, self.patch_size, self.n_masked)
        masked_patch, mask = apply_n2v_masking(patch, coords, radius=self.neighbor_radius)

        target = patch
        return masked_patch, target, mask


# ---------------------------
# 5) U-Net (depth=2)
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

        # -----------------
        # Decoder stage 1
        # -----------------
        dec1 = self.upconv1(mid)
        enc1_c = center_crop(enc1, dec1)   # ✅ 핵심
        dec1 = torch.cat([dec1, enc1_c], dim=1)
        dec1 = self.dec1_2(self.dec1_1(dec1))

        # -----------------
        # Decoder stage 0
        # -----------------
        dec0 = self.upconv2(dec1)
        enc0_c = center_crop(enc0, dec0)   # ✅ 핵심
        dec0 = torch.cat([dec0, enc0_c], dim=1)
        dec0 = self.dec0_2(self.dec0_1(dec0))

        return self.out_conv(dec0)



# ---------------------------
# 6) Masked MSE loss
# ---------------------------
def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = mask.repeat(1, pred.shape[1], 1, 1)

    mse_map = (pred - target) ** 2
    masked = mse_map * mask
    return masked.sum() / (mask.sum() + eps)


# ---------------------------
# 7) LR getter (works with schedulers too)
# ---------------------------
def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


# ---------------------------
# 8) Train (steps-based) with tqdm modes
# ---------------------------
def train_n2v_steps(
    train_root: str,
    grayscale: bool = True,
    patch_size: int = 64,
    n_masked: int = 64,
    neighbor_radius: int = 5,
    base_channels: int = 32,
    batch_size: int = 128,
    lr: float = 4e-4,
    weight_decay: float = 1e-5,
    steps: int = 5000,
    log_every: int = 100,
    ckpt_path: str = "./n2v_self_ckpt.pth",
    seed: int = 0,
    pbar_mode: str = "loss",          # "loss" | "avg_loss" | "lr"
    lr_in_desc: bool = True,          # pbar_mode="lr"일 때 lr을 postfix에만 둘지, desc에도 섞을지
) -> nn.Module:

    # seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    in_ch = 1 if grayscale else 3

    dataset = N2VPatchDataset(
        image_root_or_list=train_root,
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
        drop_last=False,  # ✅ CBSD68(68장) + batch=128이면 True면 배치 0개가 될 수 있음
        pin_memory=torch.cuda.is_available(),
    )

    if len(loader) == 0:
        raise RuntimeError(
            f"DataLoader has 0 batches. "
            f"len(dataset)={len(dataset)}, batch_size={batch_size}, drop_last={loader.drop_last}. "
            f"Fix by setting drop_last=False or reducing batch_size."
        )

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

    # tqdm
    base_desc = "Train(N2V)"
    pbar = tqdm(range(1, steps + 1), desc=base_desc, ncols=110)

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

        # ---------------------------
        # Progress bar display modes
        # ---------------------------
        if pbar_mode == "loss":
            # (1) 진행바에 loss만
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        elif pbar_mode == "avg_loss":
            # (2) 진행바에 avg_loss만 (log_every 기준 평균)
            if step % log_every == 0:
                avg = running / log_every
                running = 0.0
                pbar.set_postfix(avg_loss=f"{avg:.6f}")
            else:
                # avg 모드에서 중간 step에는 postfix를 비우거나, 직전 avg 유지가 깔끔함
                # 여기서는 "아무것도 안 바꾸는" 방식으로 깔끔 유지
                pass

        elif pbar_mode == "lr":
            # (3) lr까지 같이
            # - postfix에는 lr만 넣고,
            # - loss/avg_loss는 desc에 섞어서 보고 싶으면 lr_in_desc=True
            pbar.set_postfix(lr=f"{cur_lr:.2e}")

            if lr_in_desc:
                # desc에 metric을 넣는 방식(원하면 loss 또는 avg_loss 선택)
                # 여기서는 step_loss를 desc에 넣는 기본값
                pbar.set_description(f"{base_desc} | loss={loss.item():.4f}")
            else:
                # desc는 고정, postfix에 lr만 유지
                pbar.set_description(base_desc)

        else:
            raise ValueError('pbar_mode must be one of: "loss", "avg_loss", "lr"')

        # avg_loss 모드에서 log_every마다 running 초기화는 위에서만 함
        # 다른 모드에서는 running을 계속 쌓아도 되지만, 메모리 문제는 없고 의미만 없음
        # 그래서 log_every마다 정리하고 싶으면 아래를 켜도 됨:
        if pbar_mode != "avg_loss" and (step % log_every == 0):
            running = 0.0

    torch.save({
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
    }, ckpt_path)

    print(f"\nSaved checkpoint -> {ckpt_path}")
    return model


# ---------------------------
# 9) Load checkpoint
# ---------------------------
def load_n2v_checkpoint(
    ckpt_path: str,
    device_override: Optional[torch.device] = None,
) -> nn.Module:
    dev = device_override if device_override is not None else device
    ckpt = torch.load(ckpt_path, map_location=dev)

    in_ch = ckpt.get("in_ch", 1)
    base_channels = ckpt.get("base_channels", 32)

    model = UNet(in_channels=in_ch, base_channels=base_channels, out_channels=in_ch, kernel_size=3).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint -> {ckpt_path} (device={dev})")
    return model


# ---------------------------
# 10) Denoise + PSNR
# ---------------------------
def pad_to_multiple(x: torch.Tensor, mult: int = 4):
    # x: [B,C,H,W]
    B, C, H, W = x.shape
    pad_h = (mult - (H % mult)) % mult
    pad_w = (mult - (W % mult)) % mult

    # pad: (left, right, top, bottom)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, (H, W)

@torch.no_grad()
def denoise_image(model: nn.Module, img_path: str, grayscale: bool = True, dev: Optional[torch.device] = None) -> np.ndarray:
    dev = dev if dev is not None else next(model.parameters()).device

    x = load_image(img_path, grayscale=grayscale).unsqueeze(0).to(dev)  # [1,C,H,W]
    H0, W0 = x.shape[2], x.shape[3]

    # ✅ 4의 배수로 패딩 (pool 2번이면 mult=4)
    x_pad, (H_orig, W_orig) = pad_to_multiple(x, mult=4)

    model.eval()
    pred = model(x_pad).clamp(0, 1)

    # ✅ 원래 크기로 다시 크롭
    pred = pred[:, :, :H_orig, :W_orig]

    pred = pred[0].detach().cpu().numpy()  # [C,H,W]

    if pred.shape[0] == 1:
        pred = pred[0]  # [H,W]
    else:
        pred = pred.transpose(1, 2, 0)  # [H,W,C]

    return (pred * 255.0).astype(np.uint8)



def psnr_uint8(pred_u8: np.ndarray, gt_u8: np.ndarray, data_range: float = 255.0, eps: float = 1e-12) -> float:
    pred = pred_u8.astype(np.float64)
    gt = gt_u8.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse < eps:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def run_test_and_evaluate(
    model: nn.Module,
    test_root: str,
    out_dir: str = "./test_outputs",
    grayscale: bool = True,
    has_gt: bool = True,
    noisy_subdir: str = "noisy",
    clean_subdir: str = "clean",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    noisy_dir = os.path.join(test_root, noisy_subdir)
    if not os.path.isdir(noisy_dir):
        raise RuntimeError(f"noisy dir not found: {noisy_dir}")

    noisy_paths = list_images(noisy_dir)
    if len(noisy_paths) == 0:
        raise RuntimeError(f"No images found in: {noisy_dir}")

    clean_dir = os.path.join(test_root, clean_subdir) if has_gt else None
    if has_gt and (clean_dir is None or not os.path.isdir(clean_dir)):
        raise RuntimeError(f"clean dir not found: {clean_dir}")

    psnr_list: List[float] = []

    for npath in tqdm(noisy_paths, desc="Test(Denoise)", ncols=110):
        fname = os.path.basename(npath)

        out_u8 = denoise_image(model, npath, grayscale=grayscale)
        save_path = os.path.join(out_dir, fname)
        Image.fromarray(out_u8).save(save_path)

        if has_gt:
            cpath = os.path.join(clean_dir, fname)
            if os.path.exists(cpath):
                gt_img = Image.open(cpath)
                gt_img = gt_img.convert("L") if grayscale else gt_img.convert("RGB")
                gt_u8 = np.array(gt_img, dtype=np.uint8)
                psnr_list.append(psnr_uint8(out_u8, gt_u8))
            else:
                print(f"[WARN] GT not found: {fname}")

    print("\n====================")
    print("TEST DONE")
    print(f"Saved outputs to: {out_dir}")

    if has_gt and len(psnr_list) > 0:
        print(f"PSNR mean: {float(np.mean(psnr_list)):.4f} dB")
        print(f"PSNR std : {float(np.std(psnr_list)):.4f} dB")
        print(f"PSNR min/max: {float(np.min(psnr_list)):.4f} / {float(np.max(psnr_list)):.4f} dB")
    elif has_gt:
        print("PSNR: No matched GT pairs were found. (check filename matching rules)")
    else:
        print("PSNR: skipped (no GT provided)")
    print("====================\n")

def center_crop(enc_feat, target_feat):
    _, _, H, W = target_feat.shape
    encH, encW = enc_feat.shape[2:]
    dy = (encH - H) // 2
    dx = (encW - W) // 2
    return enc_feat[:, :, dy:dy+H, dx:dx+W]


# ---------------------------
# 11) Main (CBSD68)
# ---------------------------
if __name__ == "__main__":

    # =========================================
    # ✅ CBSD68 경로 고정
    # /home/work/N2V/CBSD68/
    #   ├── original_png/   (GT clean)
    #   └── noisy25/        (noisy sigma=25)
    # =========================================
    CBSD68_ROOT = "/home/work/N2V/CBSD68"
    TRAIN_NOISY_DIR = os.path.join(CBSD68_ROOT, "noisy25")
    TEST_ROOT = CBSD68_ROOT
    NOISY_SUBDIR = "noisy25"
    CLEAN_SUBDIR = "original_png"

    # =========================================
    # ✅ 실험 설정
    # =========================================
    grayscale = True
    patch_size = 64
    n_masked = 64
    neighbor_radius = 5

    base_channels = 32
    batch_size = 128
    lr = 4e-4
    weight_decay = 1e-5

    steps = 5000
    log_every = 100
    seed = 0

    ckpt_path = "./n2v_cbsd68_noisy25.pth"
    out_dir = "./cbsd68_noisy25_outputs"

    # =========================================
    # ✅ tqdm 표시 모드 선택 (요구한 3가지)
    #   "loss"     : (1) 진행바에 loss만
    #   "avg_loss" : (2) 진행바에 avg_loss만
    #   "lr"       : (3) lr까지 같이
    # =========================================
    PBAR_MODE = "loss"        # <- 여기만 바꿔서 모드 변경
    LR_IN_DESC = True         # PBAR_MODE="lr"일 때 loss를 desc에 같이 보여줄지

    DO_TRAIN = False
    DO_TEST = True

    if DO_TRAIN:
        print("\n====================")
        print("TRAIN: Noise2Void on CBSD68 noisy25 (noisy-only)")
        print("====================")
        print("train_dir:", TRAIN_NOISY_DIR)
        print("device   :", device)
        print("pbar_mode:", PBAR_MODE)
        print("====================\n")

        model = train_n2v_steps(
            train_root=TRAIN_NOISY_DIR,
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

    if DO_TEST:
        print("\n====================")
        print("TEST: Denoise + PSNR on CBSD68 (noisy25 vs original_png)")
        print("====================")
        print("test_root :", TEST_ROOT)
        print("noisy_sub :", NOISY_SUBDIR)
        print("clean_sub :", CLEAN_SUBDIR)
        print("out_dir   :", out_dir)
        print("====================\n")

        model = load_n2v_checkpoint(ckpt_path)

        run_test_and_evaluate(
            model=model,
            test_root=TEST_ROOT,
            out_dir=out_dir,
            grayscale=grayscale,
            has_gt=True,
            noisy_subdir=NOISY_SUBDIR,
            clean_subdir=CLEAN_SUBDIR,
        )
