# ================================================================
# TF-OFFICIAL-EXACT C-BSN (PyTorch FINAL, LDCT SAFE + CHECKPOINT)
# ================================================================

import os, glob, random, argparse # os: 경로 및 디렉토리 관리, glob: 파일 경로 조회, argparse: CLI에서 argument 넘겨 받기
import numpy as np
from tqdm import tqdm # progress bar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # PIL 이미지 로드 시 손상된 이미지도 로드하도록 설정

# ================================================================
# Device
# ================================================================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("🔥 Device:", device)

# ================================================================
# Checkpoint config (LDCT SAFE 전용)
# ================================================================
CKPT_DIR_NAME  = "ckpt_cbsn_ldct_safe"
CKPT_FILE_NAME = "C_BSN_LDCT_ckpt.pt"

# ================================================================
# Utils
# ================================================================
def torch_normalize_ct(x): # 논문에서는 정규화를 수행하지 않지만, 입력 이미지에 대한 정규화는 수행하도록 함.
    return x / 255.0

def torch_augmentation(x, seed): # 데이터 증강 함수, 시드를 이용해 회전 및 뒤집기 수행  
    # Dataset에서 unsqeeze(0)를 했으므로 차원은 (1, C, H, W)
    torch.manual_seed(seed)
    k = seed % 4
    x = torch.rot90(x, k=k, dims=[2, 3])  # H, W 축 기준으로 k번 회전
    if (seed // 4) % 2:
        x = torch.flip(x, dims=[3]) # W 축 기준으로 상하반전
    return x

def stop_grad(x): # l_inv에서 anchor로 사용되는 텐서에 대해 gradient가 가지 않도록 하는 함수
    return x.detach()

def pad_to_multiple(x, s): # CT 이미지에 대해 stride의 배수로 패딩을 수행하는 함수
    B, C, H, W = x.shape
    pad_h = (s - H % s) % s
    pad_w = (s - W % s) % s
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

# ================================================================
# TF VarianceScaling (fan_in, scale=2), convolution weight 초기화
# ================================================================
def tf_variance_scaling_(w):
    kH, kW = w.shape[2], w.shape[3] # w shape: (out_channels, in_channels, kH, kW), conv에서는 (in_channels, out_channels, kH, kW)인데 그 안에 있는 weight는 (out_channels, in_channels, kH, kW)
    # fan_in 계산 -> 논문 구현에서 사용한 방식으로, fan_in은 입력 채널 수 * 커널 높이 * 커널 너비
    fan_in = w.shape[1] * kH * kW
    std = (2.0 / fan_in) ** 0.5 # # He initialization 기법
    with torch.no_grad():
        w.normal_(0.0, std)

# ================================================================
# Random Subsampler (NO gradient)
# ================================================================
class RandomSubsampler(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.s = stride

    def forward(self, x):
        with torch.no_grad(): # Downsampling의 경우, gradient가 필요 없으므로 no_grad 사용
            B, C, H, W = x.shape
            s = self.s
            x = x.view(B, C, H//s, s, W//s, s)
            ih = torch.randint(0, s, (B,1,H//s,1,W//s,1), device=x.device)
            iw = torch.randint(0, s, (B,1,H//s,1,W//s,1), device=x.device)
            out = x.gather(3, ih.expand(-1,C,-1,-1,-1,s)) # [B, C, H//s, s, W//s, s] -> gather(3, ...) -> [B, C, H//s, 1, W//s, s]
            out = out.gather(5, iw.expand(-1,C,-1,-1,-1,-1)) # [B, C, H//s, 1, W//s, s] -> gather(5, ...) -> [B, C, H//s, 1, W//s, 1]
            return out.squeeze(5).squeeze(3) # 최종 shape: [B, C, H//s, W//s]

# ================================================================
# Space2Batch / Batch2Space
# ================================================================
def space2batch(x, s):
    B, C, H, W = x.shape
    assert H % s == 0 and W % s == 0, "patch must be divisible by stride_b"
    x = x.view(B, C, H//s, s, W//s, s) # [B, C, H, W] -> [B, C, H//s, s, W//s, s]
    x = x.permute(0,3,5,1,2,4) # [B, C, H//s, s, W//s, s] -> [B, s, s, C, H//s, W//s]
    return x.reshape(B*s*s, C, H//s, W//s) # [B, s, s, C, H//s, W//s] -> [B*s*s, C, H//s, W//s]

def batch2space(x, s, B):
    _, C, H, W = x.shape
    x = x.view(B, s, s, C, H, W) # [B*s*s, C, H//s, W//s] -> [B, s, s, C, H//s, W//s]
    x = x.permute(0,3,4,1,5,2) # [B, s, s, C, H//s, W//s] -> [B, C, H//s, s, W//s, s]
    return x.reshape(B, C, H*s, W*s) # [B, C, H//s, s, W//s, s] -> [B, C, H, W]

# ================================================================
# Masked Convolution
# ================================================================
class MaskedConv2d(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.k = k # 커널 크기
        self.pad = (k // 2) * dilation # 패딩 크기 계산
        self.dilation = dilation # dilation 설정

        self.weight = nn.Parameter(torch.empty(cout, cin, k, k)) # convolution weight, shape: (out_channels, in_channels, kH, kW)
        self.center_weight = nn.Parameter(torch.empty(cout, cin, 1, 1)) # 마스크 여부에 따른 중앙 가중치, shape: (out_channels, in_channels, 1, 1)
        self.bias = nn.Parameter(torch.zeros(cout))

        tf_variance_scaling_(self.weight) # convolution weight kaiming 초기화 수행
        tf_variance_scaling_(self.center_weight)

    def forward(self, x, is_masked):
        w = self.weight.clone()
        if is_masked:
            w[:, :, self.k//2, self.k//2] = 0.0 # is_maksed가 True인 경우, 중앙 weight를 0으로 설정
        else:
            w[:, :, self.k//2, self.k//2] = \
                self.center_weight.squeeze(-1).squeeze(-1) # is_masked가 False인 경우, 별도의 중앙 weight로 대체, [out, in 1, 1]]
        return F.relu(
            # nn.Conv2d의 경우, layer를 의미하는데 이때, 그 안에 파라미터는 공유하기 어려움, F.conv2d는 함수로 직접 weight, bias를 넣어주면 되며 가중치를 공유할 수 있음
            F.conv2d(x, w, self.bias, padding=self.pad, dilation=self.dilation)
        )

# ================================================================
# 1x1 Conv / DCM
# ================================================================
class Conv1x1(nn.Module):
    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1)
        self.act = nn.ReLU(inplace=False) if act else nn.Identity()
        tf_variance_scaling_(self.conv.weight) # convolution weight 초기화
        nn.init.zeros_(self.conv.bias) # bias는 0으로 초기화

    def forward(self, x):
        return self.act(self.conv(x))

class DCM(nn.Module): # # Dilated Convolution Module로 논문에서 사용한 DCM 구현, AP-BSN에서 상요한 module과 동일함
    def __init__(self, c, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(c, c, 1)
        tf_variance_scaling_(self.conv1.weight) # convolution weight 초기화
        tf_variance_scaling_(self.conv2.weight) # convolution weight 초기화
        nn.init.zeros_(self.conv1.bias) # bias는 0으로 초기화
        nn.init.zeros_(self.conv2.bias) # bias는 0으로 초기화

    def forward(self, x):
        f = F.relu(self.conv1(x))
        f = F.relu(self.conv2(f))
        return x + f # residual connection

# ================================================================
# Branch / CBSN
# ================================================================
class BranchTF(nn.Module): # AP-BSN에서 내부 filters는 모두 128로 고정
    def __init__(self, filters, k, dilation, num_module):
        super().__init__()
        self.masked = MaskedConv2d(filters, filters, k, 1) # # maskedconv에서는 dilation=1로 설정해서 수행
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
            x = checkpoint(dcm, x) # forward에서 gradient를 계산할 때, 메모리를 절약하기 위해 checkpointing 사용
        return self.c3(x)

class CBSN(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Conv1x1(1, 128) # LDCT의 경우, in_channel = 1, masked conv 이전에 1x1 conv 수행
        self.b1 = BranchTF(128, 3, 2, 9) # 첫 번째 브랜치로 3x3 커널, dilation=2, 9개의 DCM 모듈 사용
        self.b2 = BranchTF(128, 5, 3, 9) # 두 번째 브랜치로 5x5 커널, dilation=3, 9개의 DCM 모듈 사용
        self.f1 = Conv1x1(256, 128) # 두 브랜치의 출력을 합친 후 1x1 conv 수행
        self.f2 = Conv1x1(128, 64) # 1x1 conv 수행
        self.f3 = Conv1x1(64, 64) # 1x1 conv 수행
        self.out = Conv1x1(64, 1, act=False) # 최종 출력 채널은 1, 활성화 함수는 사용하지 않음

    def forward(self, x, is_masked):
        f = self.head(x)
        x = torch.cat( # 두 브랜치의 출력을 채널 차원에서 연결
            [self.b1(f, is_masked), self.b2(f, is_masked)],
            dim=1
        )
        return self.out(self.f3(self.f2(self.f1(x))))

# ================================================================
# Dataset
# ================================================================
class LDCTDataset(Dataset):
    def __init__(self, root, patch):
        self.files = sorted(
            glob.glob(os.path.join(root, '**', '*.png'), recursive=True) +
            glob.glob(os.path.join(root, '**', '*.PNG'), recursive=True)
        )
        assert len(self.files) > 0, "❌ No LDCT images found"
        self.patch = patch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        '''
        __getitem__의 process:
        1) 이미지 로드 (L 모드, 흑백)
        2) 텐서 변환 및 차원 추가 (C, H, W) -> (1, C, H, W)
        3) 패치 크기보다 작을 경우, 반사 패딩 수행
        4) 랜덤하게 패치 크기만큼 자르기
        5) 데이터 증강 수행 (회전 및 뒤집기)
        6) 정규화 수행
        7) 차원 축소 및 반환 (C, H, W)
        '''
        img = Image.open(self.files[idx]).convert("L")
        img = torch.from_numpy(np.array(img, np.float32))[None]
        H, W = img.shape[-2:]
        if H < self.patch or W < self.patch:
            img = F.pad(
                img,
                (0, max(0, self.patch-W), 0, max(0, self.patch-H)),
                mode="reflect"
            )
        t = random.randint(0, img.shape[-2]-self.patch)
        l = random.randint(0, img.shape[-1]-self.patch)
        img = img[:, t:t+self.patch, l:l+self.patch][None]
        return torch_normalize_ct(
            torch_augmentation(img, idx)
        ).squeeze(0)

# ================================================================
# Training
# ================================================================
def train(args):
    ckpt_dir  = os.path.join(os.getcwd(), CKPT_DIR_NAME)
    ckpt_path = os.path.join(ckpt_dir, CKPT_FILE_NAME)
    os.makedirs(ckpt_dir, exist_ok=True)

    model = CBSN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    rs    = RandomSubsampler(args.stride_i).to(device)

    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        print(f"🔁 Resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"➡️ Resumed at step {start_step}")

    loader = DataLoader(
        LDCTDataset(args.train_data_dir, args.patch),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    it = iter(loader)

    for step in tqdm(range(start_step, args.max_iter),
                     desc="🔥 C-BSN LDCT SAFE"):
        try:
            img = next(it).to(device)
        except StopIteration:
            it = iter(loader)
            img = next(it).to(device)

        img_b = pad_to_multiple(img, args.stride_b)
        pd = space2batch(img_b, args.stride_b)
        out_blind = batch2space(
            model(pd, True),
            args.stride_b,
            img_b.size(0)
        )
        l_blind = F.l1_loss(out_blind, img_b)

        if step < 200_000:
            loss = l_blind
        else:
            img_i = pad_to_multiple(img, args.stride_i)
            l_self = F.l1_loss(model(img_i, False), img_i)
            with torch.no_grad():
                target = stop_grad(model(rs(img_i), True))
            pred = rs(model(img_i, False))
            loss = l_blind + l_self + args.lambda_inv * F.l1_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10000 == 0 and step > 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "args": vars(args)
                },
                ckpt_path
            )
            print(f"💾 Saved checkpoint @ step {step}")

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_data_dir", type=str, default= "/home/work/LDCT/Sharp Kernel (D45) quater")
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--stride_b", type=int, default=5)
    p.add_argument("--stride_i", type=int, default=2)
    p.add_argument("--lambda_inv", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_iter", type=int, default=500000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    train(args)
