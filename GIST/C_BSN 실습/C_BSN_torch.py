# ================================================================
# TF-OFFICIAL-EXACT C-BSN (PyTorch FINAL + CHECKPOINT)
# - Bit-faithful loss schedule
# - Exact gradient semantics
# - Conditional blind-spot
# - RS / S2B / B2S identical to TF graph
# - Checkpoint save / resume / eval ready
# ================================================================

import os, glob, random, argparse # os: 경로, 폴더 생성, glob: 파일 검색, argparse: CLI Argument 받기
import numpy as np
from tqdm import tqdm # 학습 progess bar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # 손상된 이미지 로드 허용, 오류 방지

# ================================================================
# Device
# ================================================================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("🔥 Device:", device)

CKPT_NAME = "C_BSN_SIDD_ckpy.pt"  # 체크포인트 파일 이름


# unsqueeze(0) -> (C, H, W) -> (1, C, H, W), 0이 앞에다 추가, 1이 뒤에다 추가
# squeeze(0) -> (1, C, H, W) -> (C, H, W), 특정차원을 없애고 싶으로 함수 안에 index를 넣어주면 됨, 아니면 차원이 1인 모든 차원을 없앰
# view() -> tensor의 shape 변경, reshape과 유사하지만, view는 메모리 상에서 연속적인 경우에만 사용 가능
# permute() -> 차원 순서 변경
# reshape() -> tensor의 shape 변경, view와 유사하지만, 메모리 상에서 연속적이지 않아도 사용 가능 + 복사 가능하기에 새로운 tensor 생성
# 보통 view() -> reshape() 순으로 사용 권장
# gather() -> 특정 차원에서 인덱스를 기반으로 값을 선택하는 함수, 인덱스 텐서를 사용하여 원하는 위치의 값을 추출


# ================================================================
# Utils
# ================================================================
def torch_normalize(x): # 일단, 논문에서 정규화를 하지 않는다고 했어도 입력 이미지에 대한 정규화는 필요, 0~255 -> 0~1
    return x / 255.0

def torch_augmentation(x, seed): # Dataset에서 unsqeeze(0)를 했으므로 차원은 (1, C, H, W)
    torch.manual_seed(seed)
    k = seed % 4
    x = torch.rot90(x, k=k, dims=[2, 3]) # H, W 축 기준으로 회전
    if (seed // 4) % 2: 
        x = torch.flip(x, dims=[3]) # 수평 뒤집기
    return x

def stop_grad(x): # 본 논문에서, l_inv 계산 시 ahchor에는 gradient가 가지 않도록 detach() 사용
    return x.detach() # .detach()는 requires_grad=True인 텐서에서 gradient 계산을 멈추게 함, shpae에 변화 없음

# ================================================================
# TF VarianceScaling (fan_in, scale=2), convolution weight 초기화
# ================================================================
def tf_variance_scaling_(w): # TensorFlow의 VarianceScaling 초기화 (fan_in, scale=2)
    kH, kW = w.shape[2], w.shape[3] # w shape: (out_channels, in_channels, kH, kW), conv에서는 (in_channels, out_channels, kH, kW)인데 그 안에 있는 weight는 (out_channels, in_channels, kH, kW)
    # fan_in 계산 -> 논문 구현에서 사용한 방식으로, fan_in은 입력 채널 수 * 커널 높이 * 커널 너비
    fan_in = w.shape[1] * kH * kW
    std = (2.0 / fan_in) ** 0.5 # He initialization 기법
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
        with torch.no_grad(): # RS의 경우, 그래디언트를 계산하지 않음
            B, C, H, W = x.shape
            s = self.s
            x = x.view(B, C, H//s, s, W//s, s)

            ih = torch.randint(0, s, (B,1,H//s,1,W//s,1), device=x.device) # 각 블록마다 뽑을 (h_offset, w_offset)을 랜덤하게 선택
            iw = torch.randint(0, s, (B,1,H//s,1,W//s,1), device=x.device) # 각 블록마다 뽑을 (h_offset, w_offset)을 랜덤하게 선택

            out = x.gather(3, ih.expand(-1,C,-1,-1,-1,s)) #[B, C, H//s, s, W//s, s] -> gather(3, ...) -> [B, C, H//s, 1, W//s, s]
            out = out.gather(5, iw.expand(-1,C,-1,-1,-1,-1)) #[B, C, H//s, 1, W//s, s] -> gather(5, ...) -> [B, C, H//s, 1, W//s, 1]
            return out.squeeze(5).squeeze(3) # squeeze -> [B, C, H//s, W//s]

# ================================================================
# Space2Batch / Batch2Space (gradient ON)
# ================================================================
def space2batch(x, s):
    B, C, H, W = x.shape
    assert H % s == 0 and W % s == 0, "Patch must be divisible by stride_b" # space2batch는 stride_b로 나누어 떨어져야 함
    x = x.view(B, C, H//s, s, W//s, s) # [B, C, H//s, s, W//s, s]
    x = x.permute(0,3,5,1,2,4) # [B, s, s, C, H//s, W//s]
    return x.reshape(B*s*s, C, H//s, W//s) # [B*s*s, C, H//s, W//s], space2batch는 배치 기준으로 블록을 나누어 배치 크기를 늘림

def batch2space(x, s, B): # batch2space는 space2batch의 역연산, B는 원래 배치 크기
    _, C, H, W = x.shape
    x = x.view(B, s, s, C, H, W) # [B, s, s, C, H, W]
    x = x.permute(0,3,4,1,5,2) # [B, C, H, s, W, s]
    return x.reshape(B, C, H*s, W*s) # [B, C, H*s, W*s], batch2space는 배치 기준으로 블록을 합쳐 배치 크기를 줄임

# ================================================================
# Masked Convolution (TF-exact)
# ================================================================
class MaskedConv2d(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.k = k # kernel size
        self.pad = (k // 2) * dilation # padding_size 계산, TF 'SAME' padding과 동일 특히, dilation을 활용하는 경우, k // 2 * dilation으로 수행
        self.dilation = dilation # dilation

        self.weight = nn.Parameter(torch.empty(cout, cin, k, k)) # convolution weight, shape: (out_channels, in_channels, kH, kW)
        self.center_weight = nn.Parameter(torch.empty(cout, cin, 1, 1)) # masked의 경우, 중앙을 0으로 만들기 때문에 Unmasked 시에 사용할 중앙 weight
        self.bias = nn.Parameter(torch.zeros(cout)) # bias

        tf_variance_scaling_(self.weight) # convolution weight kaiming 초기화 수행
        tf_variance_scaling_(self.center_weight) # 마스크 여부에 따른 중앙 weight도 초기화

    def forward(self, x, is_masked):
        w = self.weight.clone()
        if is_masked:
            w[:, :, self.k//2, self.k//2] = 0.0  # is_maksed가 True인 경우, 중앙 weight를 0으로 설정
        else:
            w[:, :, self.k//2, self.k//2] = \
                self.center_weight.squeeze(-1).squeeze(-1)  # is_masked가 False인 경우, 별도의 중앙 weight로 대체, [out, in 1, 1]]

        out = F.conv2d( # nn.Conv2d의 경우, layer를 의미하는데 이때, 그 안에 파라미터는 공유하기 어려움, F.conv2d는 함수로 직접 weight, bias를 넣어주면 되며 가중치를 공유할 수 있음
            x, w, self.bias,
            padding=self.pad,
            dilation=self.dilation
        )
        return F.relu(out)

# ================================================================
# 1x1 Conv
# ================================================================
class Conv1x1(nn.Module):
    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1)
        self.act = nn.ReLU(inplace=False) if act else nn.Identity()
        tf_variance_scaling_(self.conv.weight) # 1x1 convolution weight 초기화
        nn.init.zeros_(self.conv.bias) # 1x1 convolution bias 초기화

    def forward(self, x):
        return self.act(self.conv(x))

# ================================================================
# Dilated Convolution Module (DCM)
# ================================================================
class DCM(nn.Module): # Dilated Convolution Module로 논문에서 사용한 DCM 구현, AP-BSN에서 상요한 module과 동일함
    def __init__(self, c, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(c, c, 1)
        self.relu = nn.ReLU(inplace=False)

        tf_variance_scaling_(self.conv1.weight) # convoluition weight 초기화
        tf_variance_scaling_(self.conv2.weight) # convolution weight 초기화
        nn.init.zeros_(self.conv1.bias) # convolution bias 초기화
        nn.init.zeros_(self.conv2.bias) # convolution bias 초기화

    def forward(self, x):
        f = self.relu(self.conv1(x))
        f = self.relu(self.conv2(f))
        return x + f # skip connction

# ================================================================
# Branch (TF style)
# ================================================================
from torch.utils.checkpoint import checkpoint

class BranchTF(nn.Module): # AP-BSN에서 내부 filters는 모두 128로 고정
    def __init__(self, filters, k, dilation, num_module):
        super().__init__()
        self.masked = MaskedConv2d(filters, filters, k, 1) # maskedconv에서는 dilation=1로 설정해서 수행
        self.c1 = Conv1x1(filters, filters)
        self.c2 = Conv1x1(filters, filters)
        self.dcms = nn.ModuleList( # DCM 모듈을 MoudleList로 반복문으로 생성
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

# ================================================================
# CBSN Network
# ================================================================
class CBSN(nn.Module):
    def __init__(self, in_channels=3, filters=128, num_module=9): # in_channels=3 (RGB), filters=128 (논문에서 고정), num_module=9 (논문에서 고정)
        super().__init__()
        self.head = Conv1x1(in_channels, filters) # maske conv 전에 1x1 conv로 채널 수 맞춤
        self.b1 = BranchTF(filters, 3, 2, num_module)# 첫 번째 브랜치: 커널 크기 3, dilation 2, num_module 개수
        self.b2 = BranchTF(filters, 5, 3, num_module)# 두 번째 브랜치: 커널 크기 5, dilation 3, num_module 개수
        self.f1 = Conv1x1(filters*2, filters) # 두 브랜치 출력 채널을 합쳐서 1x1 conv 수행
        self.f2 = Conv1x1(filters, 64) # 1x1 conv로 채널 수 64로 감소
        self.f3 = Conv1x1(64, 64) # 1x1 conv로 채널 수 64로 유지
        self.out = Conv1x1(64, in_channels, act=False) # 최종 출력 1x1 conv, act=False로 활성화 함수 없음

    def forward(self, x, is_masked):
        f = self.head(x)
        b1 = self.b1(f, is_masked)
        b2 = self.b2(f, is_masked)
        x = torch.cat([b1, b2], dim=1) # 두 브랜치 출력 concatenate 수행, dim=1은 채널 축 기준
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.out(x)

# ================================================================
# Dataset (SIDD noisy only)
# ================================================================
class SIDDDataset(Dataset):
    def __init__(self, root, patch=240):
        self.files = sorted(
            glob.glob(os.path.join(root, '*', 'NOISY_SRGB_*.png')) +
            glob.glob(os.path.join(root, '*', 'NOISY_SRGB_*.PNG'))
        )
        assert len(self.files) > 0, "❌ No SIDD images found"
        self.patch = patch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        '''
        __getitem__의 process:
        1. 이미지 로드 및 RGB 변환 - PIL 사용
        2. NumPy 배열로 변환 및 float32 타입으로 캐스팅 & 채널 우선 형식으로 전치 (C, H, W)
        3. 무작위로 패치 추출 (patch 크기)
        4. Patch를 PyTorch 텐서로 변환 및 배치 차원 추가
        5. 데이터 증강 (무작위 회전 및 뒤집기)
        6. 정규화 (0~255 -> 0~1)
        7. 배치 차원 제거 및 반환
        8. 최종 반환 형태: (C, patch, patch)
        '''
        img = Image.open(self.files[idx]).convert("RGB")
        img = np.array(img).astype(np.float32).transpose(2,0,1)
        _, H, W = img.shape
        t = random.randint(0, H-self.patch)
        l = random.randint(0, W-self.patch)
        patch = torch.from_numpy(
            img[:, t:t+self.patch, l:l+self.patch]
        ).unsqueeze(0)
        patch = torch_augmentation(patch, idx)
        return torch_normalize(patch).squeeze(0)

# ================================================================
# Training
# ================================================================
def train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(args.ckpt_dir) # TensorBoard SummaryWriter 생성

    model = CBSN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)

    start_step = 0
    ckpt_path = os.path.join(args.ckpt_dir, CKPT_NAME)

    # Checkpoint resume
    if args.resume and os.path.exists(ckpt_path):
        print(f"🔁 Resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"➡️ Resumed at step {start_step}")

    rs = RandomSubsampler(args.stride_i).to(device) # RandomSubsampler 생성 및 디바이스 할당

    # DataLoader
    loader = DataLoader(
        SIDDDataset(args.train_data_dir, args.patch),
        batch_size=1, shuffle=True, num_workers=4
    )

    it = iter(loader)
    warmup = 200_000

    for step in tqdm(range(start_step, args.max_iter),
                     desc="🔥 TF-OFFICIAL-EXACT C-BSN"):
        try:
            img = next(it).to(device)
        except StopIteration:
            it = iter(loader)
            img = next(it).to(device)

        # Blind-spot loss
        pd = space2batch(img, args.stride_b)
        out_blind = batch2space(
            model(pd, is_masked=True),
            args.stride_b,
            img.size(0)
        )
        l_blind = F.l1_loss(out_blind, img)

        # Total loss
        # 학습 초기에는 Warm-up을 활용해서 l_blind만 사용
        if step < warmup:
            loss = l_blind
        else:
            out = model(img, is_masked=False)
            l_self = F.l1_loss(out, img)

            with torch.no_grad():
                ds_img = rs(img) # downsampled image의 경우, gradient가 필요 없으므로 with torch.no_grad() 사용

            target = stop_grad(model(ds_img, is_masked=True)) # anchor는 gradient가 가지 않도록 stop_grad 사용, l_inv 계산 시에만 사용
            pred = rs(model(img, is_masked=False))

            l_inv = F.l1_loss(pred, target)
            loss = l_blind + l_self + args.lambda_inv * l_inv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/blind", l_blind.item(), step)
        writer.add_scalar("Loss/total", loss.item(), step)
        if step >= warmup:
            writer.add_scalar("Loss/self", l_self.item(), step)
            writer.add_scalar("Loss/inv", l_inv.item(), step)

        if step % 10000 == 0 and step > 0:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args)
                },
                ckpt_path
            )
            print(f"💾 Saved checkpoint @ step {step}")

    writer.close()

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int, default=500000)
    parser.add_argument("--stride_b", type=int, default=5)
    parser.add_argument("--stride_i", type=int, default=2)
    parser.add_argument("--lambda_inv", type=float, default=2.0)
    parser.add_argument("--patch", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(args)



# python C_BSN_self.py \
#   --train_data_dir /home/work/AP-BSN/SIDD_Small_sRGB_Only/Data \
#   --ckpt_dir ./ckpt_cbsn_tf_exact \
#   --lr 1e-4 \
#   --max_iter 500000 \
#   --stride_b 5 \
#   --stride_i 2 \
#   --lambda_inv 2.0 \
#   --patch 240