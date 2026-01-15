import os
import glob
import random
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


# ----------------------------------------
# 1. 재현성 고정
# ----------------------------------------
def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# ======================================================================================
# 2. Pixel-Shuffle Down / Up (Asymmetric PD)
# ======================================================================================
def pd_down(x, f, pad):
    B, C, H, W = x.shape
    subs = []
    for i in range(f):
        for j in range(f):
            sub = x[:, :, i::f, j::f]
            if pad > 0:
                sub = F.pad(sub, (pad, pad, pad, pad), mode="reflect")
            subs.append(sub)
    return torch.cat(subs, dim=0)


def pd_up(x, f, pad):
    Bff, C, h, w = x.shape
    B = Bff // (f * f)

    if pad > 0:
        x = x[:, :, pad:-pad, pad:-pad]
        h -= 2 * pad
        w -= 2 * pad

    out = torch.zeros((B, C, h * f, w * f), device=x.device)

    idx = 0
    for i in range(f):
        for j in range(f):
            out[:, :, i::f, j::f] = x[idx * B:(idx + 1) * B]
            idx += 1
    return out


# ======================================================================================
# 3. DBSNl (Blind-Spot Network)
# ======================================================================================
class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.shape
        self.mask[:, :, kH // 2, kW // 2] = 0

    def forward(self, x):
        self.weight.data = self.weight.data * self.mask
        return super().forward(x)


class DCl(nn.Module):
    def __init__(self, stride, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=stride, dilation=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class DC_branchl(nn.Module):
    def __init__(self, stride, ch, num_module):
        super().__init__()
        layers = [
            CentralMaskedConv2d(ch, ch, kernel_size=2 * stride - 1, padding=stride - 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.ReLU(inplace=True),
        ]
        layers += [DCl(stride, ch) for _ in range(num_module)]
        layers += [nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class DBSNl(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, num_module=9):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_ch, base_ch, 1), nn.ReLU(inplace=True))
        self.b1 = DC_branchl(2, base_ch, num_module)
        self.b2 = DC_branchl(3, base_ch, num_module)
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, in_ch, 1),
        )

    def forward(self, x):
        x = self.head(x)
        x = torch.cat([self.b1(x), self.b2(x)], dim=1)
        return self.tail(x)


# ======================================================================================
# 4. APBSN Wrapper + R³
# ======================================================================================
class APBSN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pd_a = 5
        self.pd_b = 2
        self.pd_pad = 2

        self.R3_T = 8
        self.R3_p = 0.16

        self.bsn = DBSNl()

    def forward(self, x, pd=None):
        if pd is None:
            pd = self.pd_a

        if pd > 1:
            x = pd_down(x, pd, self.pd_pad)
            x = self.bsn(x)
            x = pd_up(x, pd, self.pd_pad)
        else:
            x = F.pad(x, (2, 2, 2, 2), mode="reflect")
            x = self.bsn(x)[:, :, 2:-2, 2:-2]
        return x

    @torch.no_grad()
    def denoise(self, x):
        """
        inference + R³
        """
        B, C, H, W = x.shape
        pad_h = (self.pd_b - H % self.pd_b) % self.pd_b
        pad_w = (self.pd_b - W % self.pd_b) % self.pd_b
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        base = self.forward(x, pd=self.pd_b)
        outs = []
        for _ in range(self.R3_T):
            mask = (torch.rand_like(x[:, :1]) < self.R3_p)
            tmp = base.clone()
            tmp[mask.expand_as(tmp)] = x[mask.expand_as(x)]
            tmp = F.pad(tmp, (2, 2, 2, 2), mode="reflect")
            tmp = self.bsn(tmp)[:, :, 2:-2, 2:-2]
            outs.append(tmp)

        out = torch.stack(outs).mean(0)
        return out[:, :, :H, :W]


# ======================================================================================
# 5. SIDD Pair 수집 + split
# ======================================================================================
def collect_sidd_pairs(root):
    noisy_paths = glob.glob(os.path.join(root, "**/NOISY_SRGB_*.PNG"), recursive=True)
    noisy_paths = sorted(noisy_paths)

    pairs = []
    for npath in noisy_paths:
        gpath = npath.replace("NOISY_SRGB_", "GT_SRGB_")
        if os.path.exists(gpath):
            pairs.append((npath, gpath))
        else:
            print(f"[WARN] GT not found for: {npath}")

    if len(pairs) == 0:
        raise RuntimeError("No (NOISY, GT) pairs found. Check root path.")
    return pairs


def split_pairs_5_3_2(pairs, seed=0):
    rng = random.Random(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)

    N = len(pairs)
    n_train = int(N * 0.5)
    n_val = int(N * 0.3)
    n_test = N - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"[Split] total={N}, train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


# ======================================================================================
# 6. Dataset
# ======================================================================================
class SIDDTrainNoisyPatchDataset(Dataset):
    def __init__(self, train_pairs, patch_size=120, patches_per_epoch=20000):
        super().__init__()
        self.noisy_paths = [n for (n, g) in train_pairs]
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        if len(self.noisy_paths) == 0:
            raise RuntimeError("Train noisy paths are empty.")

    def __len__(self):
        return self.patches_per_epoch

    def __getitem__(self, idx):
        img = Image.open(random.choice(self.noisy_paths)).convert("RGB")
        img = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0

        _, H, W = img.shape
        ps = self.patch_size

        if H < ps or W < ps:
            img = F.interpolate(img.unsqueeze(0), size=(max(H, ps), max(W, ps)),
                                mode="bilinear", align_corners=False).squeeze(0)
            _, H, W = img.shape

        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        patch = img[:, top:top + ps, left:left + ps]

        k = random.randint(0, 3)
        patch = torch.rot90(patch, k, dims=(1, 2))

        if random.random() < 0.5:
            patch = torch.flip(patch, dims=(2,))
        if random.random() < 0.5:
            patch = torch.flip(patch, dims=(1,))

        return patch


class SIDDPairImageDataset(Dataset):
    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs
        if len(self.pairs) == 0:
            raise RuntimeError("Pairs are empty.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        npath, gpath = self.pairs[idx]
        noisy = Image.open(npath).convert("RGB")
        gt = Image.open(gpath).convert("RGB")
        noisy = torch.from_numpy(np.array(noisy, dtype=np.float32)).permute(2, 0, 1) / 255.0
        gt = torch.from_numpy(np.array(gt, dtype=np.float32)).permute(2, 0, 1) / 255.0
        return noisy, gt, npath


# ======================================================================================
# 7. Metrics
# ======================================================================================
def psnr_torch(pred, target, eps=1e-10):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean().item()


def ssim_torch(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    def gaussian_window(ws, sigma=1.5, device="cpu", dtype=torch.float32):
        coords = torch.arange(ws, device=device, dtype=dtype) - ws // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        w = g[:, None] * g[None, :]
        return w

    B, C, H, W = pred.shape
    device = pred.device
    dtype = pred.dtype

    w = gaussian_window(window_size, device=device, dtype=dtype)
    w = w.view(1, 1, window_size, window_size).repeat(C, 1, 1, 1)

    pad = window_size // 2
    mu1 = F.conv2d(pred, w, padding=pad, groups=C)
    mu2 = F.conv2d(target, w, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, w, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, w, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, w, padding=pad, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
    )
    return ssim_map.mean(dim=(1, 2, 3)).mean().item()


# ======================================================================================
# 8. Tile inference (OOM 해결 핵심)
# ======================================================================================
@torch.no_grad()
def infer_tiled(model, x, device, tile=256, overlap=32, use_r3=False):
    """
    x: [1,C,H,W] in [0,1]
    tile/overlap: 타일 크기와 겹침(경계 seam 완화)
    use_r3=False: val (no-R3) -> forward(pd_b)
    use_r3=True : test (R3)   -> denoise()
    """
    assert x.dim() == 4 and x.size(0) == 1
    _, C, H, W = x.shape

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile must be larger than overlap")

    out = torch.zeros((1, C, H, W), device=device, dtype=x.dtype)
    weight = torch.zeros((1, 1, H, W), device=device, dtype=x.dtype)

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top + tile, H)
            right = min(left + tile, W)

            # 타일이 끝에서 작아지면, 시작점을 당겨서 항상 tile 크기 유지(가능하면)
            top0 = max(0, bottom - tile)
            left0 = max(0, right - tile)
            bottom0 = min(top0 + tile, H)
            right0 = min(left0 + tile, W)

            patch = x[:, :, top0:bottom0, left0:right0].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if use_r3:
                    pred = model.denoise(patch)
                else:
                    pred = model.forward(patch, pd=model.pd_b)

            pred = pred.clamp(0.0, 1.0)

            out[:, :, top0:bottom0, left0:right0] += pred
            weight[:, :, top0:bottom0, left0:right0] += 1.0

    out = out / weight.clamp_min(1e-8)
    return out


# ======================================================================================
# 9. Train / Eval
# ======================================================================================
def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_one_epoch(model, loader, optimizer, device, epoch=1, epochs=1):
    model.train()
    running = 0.0
    n = 0

    data_time_sum = 0.0
    iter_time_sum = 0.0

    end = time.perf_counter()
    pbar = tqdm(loader, desc=f"Train [{epoch}/{epochs}]", leave=False)

    for batch in pbar:
        data_time = time.perf_counter() - end
        data_time_sum += data_time

        start_iter = time.perf_counter()

        noisy = batch.to(device)

        pred = model(noisy, pd=model.pd_a)
        loss = F.l1_loss(pred, noisy)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        iter_time = time.perf_counter() - start_iter
        iter_time_sum += iter_time

        running += loss.item()
        n += 1

        avg_loss = running / max(1, n)
        avg_data = data_time_sum / max(1, n)
        avg_iter = iter_time_sum / max(1, n)

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "data": f"{avg_data*1000:.1f}ms",
            "iter": f"{avg_iter*1000:.1f}ms"
        })

        end = time.perf_counter()

    return running / max(1, n), data_time_sum / max(1, n), iter_time_sum / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, max_images=None, desc="Eval", use_r3=False, tile=256, overlap=32):
    """
    ✅ OOM 방지: 항상 tile inference로 평가
    - val: use_r3=False (no-R3)
    - test: use_r3=True (R3)
    """
    model.eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    start = time.perf_counter()
    pbar = tqdm(loader, desc=desc, leave=False)

    for noisy, gt, path in pbar:
        noisy = noisy.to(device)
        gt = gt.to(device)

        pred = infer_tiled(model, noisy, device, tile=tile, overlap=overlap, use_r3=use_r3)

        psnr_sum += psnr_torch(pred, gt)
        ssim_sum += ssim_torch(pred, gt)
        count += 1

        pbar.set_postfix({
            "PSNR": f"{(psnr_sum/max(1,count)):.2f}",
            "SSIM": f"{(ssim_sum/max(1,count)):.4f}"
        })

        if (max_images is not None) and (count >= max_images):
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start
    if count == 0:
        return 0.0, 0.0, elapsed
    return psnr_sum / count, ssim_sum / count, elapsed


def save_checkpoint(save_path, model, optimizer, epoch, best_psnr):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_psnr": best_psnr,
    }, save_path)


def load_checkpoint(ckpt_path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_psnr", -1.0)


# ======================================================================================
# 10. main
# ======================================================================================
@dataclass
class Config:
    sidd_root: str = "/home/work/data/SIDD_Small_sRGB_Only"
    seed: int = 0

    patch_size: int = 120
    patches_per_epoch: int = 20000

    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0

    save_dir: str = "./checkpoints_apbsn"
    save_every: int = 1

    eval_max_images: int = 50

    # ✅ OOM 대응: 평가용 타일 파라미터 (여기만 조절하면 됨)
    eval_tile: int = 256
    eval_overlap: int = 32


def main():
    cfg = Config()
    set_seed(cfg.seed)

    pairs = collect_sidd_pairs(cfg.sidd_root)
    train_pairs, val_pairs, test_pairs = split_pairs_5_3_2(pairs, seed=cfg.seed)

    train_ds = SIDDTrainNoisyPatchDataset(train_pairs, patch_size=cfg.patch_size, patches_per_epoch=cfg.patches_per_epoch)
    val_ds = SIDDPairImageDataset(val_pairs)
    test_ds = SIDDPairImageDataset(test_pairs)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    device = get_device()
    print(f"[Device] {device}")

    model = APBSN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_psnr = -1.0

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, avg_data_t, avg_iter_t = train_one_epoch(
            model, train_loader, optimizer, device, epoch=epoch, epochs=cfg.epochs
        )

        # ✅ Val: no-R3 + Tiled inference
        val_psnr, val_ssim, val_time = evaluate(
            model, val_loader, device,
            max_images=cfg.eval_max_images,
            desc=f"Val(no-R3,tiled) [{epoch}/{cfg.epochs}]",
            use_r3=False,
            tile=cfg.eval_tile,
            overlap=cfg.eval_overlap
        )

        epoch_time = time.perf_counter() - epoch_start

        print(
            f"[Epoch {epoch:03d}] "
            f"train_L1={train_loss:.6f} | "
            f"avg_data={avg_data_t*1000:.1f}ms/iter | "
            f"avg_iter={avg_iter_t*1000:.1f}ms/iter | "
            f"epoch_time={epoch_time/60:.2f}min | "
            f"val(no-R3,tiled)_PSNR={val_psnr:.3f} | val(no-R3,tiled)_SSIM={val_ssim:.4f} | "
            f"val_time={val_time/60:.2f}min"
        )

        if epoch % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(save_path, model, optimizer, epoch, best_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = os.path.join(cfg.save_dir, "best.pt")
            save_checkpoint(best_path, model, optimizer, epoch, best_psnr)
            print(f"  -> [BEST] updated! best_PSNR={best_psnr:.3f} saved to {best_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------
    # Test: ✅ R3 포함 + Tiled inference
    # -------------------------
    best_path = os.path.join(cfg.save_dir, "best.pt")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, optimizer=None, map_location=device)
        print(f"[Load] best checkpoint loaded: {best_path}")
    else:
        print(f"[WARN] best checkpoint not found. Use last model.")

    test_psnr, test_ssim, test_time = evaluate(
        model, test_loader, device,
        max_images=cfg.eval_max_images,
        desc="Test(R3,tiled)",
        use_r3=True,
        tile=cfg.eval_tile,
        overlap=cfg.eval_overlap
    )
    print(f"[TEST(R3,tiled)] PSNR={test_psnr:.3f} | SSIM={test_ssim:.4f} | time={test_time/60:.2f}min")


if __name__ == "__main__":
    main()
