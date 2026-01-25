# ================================================================
# CBSN LDCT SAFE Evaluation Script (OOM-SAFE FINAL)
# - CBSN : GPU
# - LPIPS: CPU (즉시 계산, 누적 X)
# ================================================================

import os, glob
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips

# ================================================================
# PIL SAFETY
# ================================================================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================================================================
# Device
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print("🔥 Device:", device)

# ================================================================
# Paths
# ================================================================
CKPT_PATH = "/home/work/LDCT/ckpt_cbsn_ldct_safe/C_BSN_LDCT_ckpt.pt"
QD_ROOT   = "/home/work/LDCT/Sharp Kernel (D45) quater"
FD_ROOT   = "/home/work/LDCT/Sharp Kernel (D45)"

# ================================================================
# Model (EXACT TRAIN STRUCTURE)
# ================================================================
class MaskedConv2d(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.k = k
        self.pad = (k // 2) * dilation
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(cout, cin, k, k))
        self.center_weight = nn.Parameter(torch.empty(cout, cin, 1, 1))
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x, is_masked):
        w = self.weight.clone()
        if is_masked:
            w[:, :, self.k//2, self.k//2] = 0.0
        else:
            w[:, :, self.k//2, self.k//2] = \
                self.center_weight.squeeze(-1).squeeze(-1)
        return F.relu(
            F.conv2d(x, w, self.bias,
                     padding=self.pad,
                     dilation=self.dilation)
        )

class Conv1x1(nn.Module):
    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1)
        self.act = nn.ReLU(inplace=False) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.conv(x))

class DCM(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(c, c, 1)
    def forward(self, x):
        return x + F.relu(self.conv2(F.relu(self.conv1(x))))

class BranchTF(nn.Module):
    def __init__(self, filters, k, dilation, num_module):
        super().__init__()
        self.masked = MaskedConv2d(filters, filters, k, 1)
        self.c1 = Conv1x1(filters, filters)
        self.c2 = Conv1x1(filters, filters)
        self.dcms = nn.ModuleList(
            [DCM(filters, dilation) for _ in range(num_module)]
        )
        self.c3 = Conv1x1(filters, filters)

    def forward(self, x, is_masked):
        x = self.masked(x, is_masked)
        x = self.c1(x)
        x = self.c2(x)
        for dcm in self.dcms:
            x = dcm(x)
        return self.c3(x)

class CBSN(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Conv1x1(1, 128)
        self.b1 = BranchTF(128, 3, 2, 9)
        self.b2 = BranchTF(128, 5, 3, 9)
        self.f1 = Conv1x1(256, 128)
        self.f2 = Conv1x1(128, 64)
        self.f3 = Conv1x1(64, 64)
        self.out = Conv1x1(64, 1, act=False)

    def forward(self, x):
        f = self.head(x)
        x = torch.cat(
            [self.b1(f, False), self.b2(f, False)],
            dim=1
        )
        return self.out(self.f3(self.f2(self.f1(x))))

# ================================================================
# Safe loader
# ================================================================
def load_ct(path):
    try:
        img = Image.open(path).convert("L")
        img = np.array(img, np.float32) / 255.0
        return torch.from_numpy(img)[None, None]
    except:
        print(f"⚠️ Skip corrupted: {path}")
        return None

# ================================================================
# Evaluation
# ================================================================
def main():
    # -------- CBSN (GPU) --------
    model = CBSN().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # -------- LPIPS (CPU) --------
    lpips_net = lpips.LPIPS(net="alex").to(cpu)
    lpips_vals = []

    psnr_list, ssim_list = [], []

    for pid in tqdm(sorted(os.listdir(QD_ROOT)), desc="👤 Patients"):
        qd_dir = os.path.join(QD_ROOT, pid)
        fd_dir = os.path.join(FD_ROOT, pid)
        if not os.path.isdir(fd_dir):
            continue

        qd_files = sorted(glob.glob(qd_dir + "/*.png"))
        fd_files = sorted(glob.glob(fd_dir + "/*.png"))
        min_len = min(len(qd_files), len(fd_files))

        for i in tqdm(range(min_len), leave=False, desc=f"  🖼 {pid}"):
            x  = load_ct(qd_files[i])
            gt = load_ct(fd_files[i])
            if x is None or gt is None:
                continue

            x  = x.to(device)
            gt = gt.to(device)

            with torch.no_grad():
                y = model(x).clamp(0,1)

            # ---- PSNR / SSIM ----
            y_np  = y.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            psnr_list.append(compute_psnr(gt_np, y_np, data_range=1.0))
            ssim_list.append(compute_ssim(gt_np, y_np, data_range=1.0))

            # ---- LPIPS (CPU, immediate) ----
            with torch.no_grad():
                lp = lpips_net(
                    gt.cpu().repeat(1,3,1,1),
                    y.cpu().repeat(1,3,1,1)
                ).item()
            lpips_vals.append(lp)

            torch.cuda.empty_cache()

    print("\n================ CBSN LDCT EVAL =================")
    print(f"PSNR  : {np.mean(psnr_list):.3f}")
    print(f"SSIM  : {np.mean(ssim_list):.4f}")
    print(f"LPIPS : {np.mean(lpips_vals):.4f}")
    print("================================================")

# ================================================================
if __name__ == "__main__":
    main()
