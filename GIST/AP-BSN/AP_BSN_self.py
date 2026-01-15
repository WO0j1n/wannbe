import os              # 파일/디렉토리 경로 처리용
import glob            # 패턴으로 파일을 한번에 찾기 위한 모듈
import random          # 파이썬 난수(augmentation/샘플링)에 사용
from dataclasses import dataclass  # (현재 코드에서는 사용 안 함) 설정 구조체 만들 때 주로 씀

import numpy as np

import torch           # PyTorch 핵심
import torch.nn as nn  # 신경망 레이어 모듈
import torch.nn.functional as F  # 함수형 API (pad 등)
import torch.optim as optim      # optimizer들

from torch.utils.data import Dataset, DataLoader  # 커스텀 데이터셋/로더

import torchvision
import torchvision.transforms as transforms  # (현재 코드에선 transform 객체는 안 씀)

from PIL import Image  # 이미지 로딩용



# ----------------------------------------
# 1. 재현성 고정
# ----------------------------------------

def set_seed(seed=0):
    """
    실험 재현성을 위해 모든 난수 시드 고정
    (논문 재현에서 매우 중요)
    """
    random.seed(seed)   # 파이썬 random 시드 고정
    torch.manual_seed(seed) # PyTorch CPU 시드 고정
    torch.mps.manual_seed(seed) # PyTorch GPU 시드 고정
    np.random.seed(seed)


# ======================================================================================
# 2. SIDD Noisy Patch Dataset
# ======================================================================================

class SIDDNoisyPatchDataset(Dataset):
    """
    ✔ SIDD noisy 이미지에서
    ✔ 매 iteration마다 랜덤 120x120 patch 생성
    ✔ epoch당 20,000개 patch를 '가상으로' 제공

    핵심 아이디어:
    - clean GT 없음
    - noisy image 하나만으로 학습
    - __len__ == patches_per_epoch
    """

    def __init__(self, root, patch_size=120, patches_per_epoch=20000):
        super().__init__()

        self.patch_size = patch_size # 한 번에 뽑을 패치 크기
        self.patches_per_epoch = patches_per_epoch # epoch 당 패치 수

        # SIDD noisy 이미지들 수집
        self.paths = []
        for ext in ("png", "jpg", "jpeg", "bmp"):
            self.paths += glob.glob(os.path.join(root, f"**/*.{ext}"), recursive=True)

            self.paths = sorted(self.paths)

        if len(self.paths) == 0:
            raise RuntimeError("SIDD noisy images not found") # 이미지 없으면 에러 처리

    def __len__(self):
        # 한 epoch에 몇 번 sampling할지
        return self.patches_per_epoch

    def __getitem__(self, idx):
        # idx는 DaraLoader가 주지만 여기서는 실제로 사용 한 함(랜덤 샘플링)

        # 1) noisy 이미지 하나 랜덤 선택, PIL 객체로 열어서 RGB 채널로 통일함
        img = Image.open(random.choice(self.paths)).convert("RGB")

        # 2) [H,W,3] → [3,H,W], float32 [0,1]
        # PIL 이미지 -> ByteTensor -> view 변환(H, W, 3) -> numpy로 변경 -> view 변환(3, H, W) -> float32 -> [0,1] 정규화
        img = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            .view(img.size[1], img.size[0], 3)
            .numpy()
        ).float().permute(2, 0, 1) / 255.0

        _, H, W = img.shape # Channels, Height, Width Unpacking
        ps = self.patch_size # Patch Size -> 120으로 설정

        # 3) 랜덤 크롭 좌표
        top = random.randint(0, H - ps) # patch y 시작 좌표
        left = random.randint(0, W - ps) # patch x 시작 좌표

        # top ≤ y < top+ps, left ≤ x < left+ps 범위 선택
        patch = img[:, top:top+ps, left:left+ps] # (C, H, W)일 때, C는 유지하고 H W에서 패치 크기만큼 자름

        # 4) 증강: 90도 회전
        k = random.randint(0, 3) # 0, 1, 2, 3중 하나 랜덤
        patch = torch.rot90(patch, k, dims=(1, 2)) # (H, W) 축을 기준으로 90도 k번 회전

        # 5) 증강: 수평 / 수직 flip
        if random.random() < 0.5:
            patch = torch.flip(patch, dims=(2,)) # 좌우반전, dims = (2, )은 W축 기준
        if random.random() < 0.5:
            patch = torch.flip(patch, dims=(1,)) # 상하반전, dims = (1, )은 H축 기준

        return patch # [C, ps, ps] 크기의 패치 반환


# ======================================================================================
# 3. Pixel-Shuffle Down / Up (Asymmetric PD)
# ======================================================================================

def pd_down(x, f, pad):
    """
    Asymmetric Pixel-Shuffle Downsampling

    입력:
        x: [B, C, H, W]
    출력:
        [B*f*f, C, H/f + 2p, W/f + 2p]

    의미:
    - 공간을 fxf 격자로 나눠
    - i::f, j::f 형태로 f 간격으로 띄어서 픽셀을 뽑는 서브 샘플
    - 서로 다른 서브샘플을 '배치 차원'으로 쌓아서 네트워크가 서로 벌어진 픽셀들을 보게 됨
    - 공간적 상관을 깨기 위한 핵심 연산
    """
    B, C, H, W = x.shape
    subs = [] # 서브샘플들을 담을 리스트 f * f개의 sub_image를 담음 f가 2이면 4개의 서브 샘플링, 5인 경우, 25개의 서브 샘플링

    for i in range(f):
        for j in range(f):
            sub = x[:, :, i::f, j::f]  # 오프셋 해당하는 격자 픽셀 추출
            # 경계 영향 줄이기 위해서 reflect 패딩 적용
            # padding = (left, right, top, bottom)
            if pad > 0:
                sub = F.pad(sub, (pad, pad, pad, pad), mode="reflect")

            subs.append(sub) #sub_iamge 리스트에 추가

    return torch.cat(subs, dim=0) # 배치 차원으로 쌓기 dim = 0, [B*f*f, C, H, W]


def pd_up(x, f, pad):
    """
    PD down의 정확한 역연산 -> pixel shuffle 의 역연산
    """
    Bff, C, h, w = x.shape
    B = Bff // (f * f) # 배치 차원으로 쌓아둔 것을 원래 배치 크기로 복구

    if pad > 0: 
        x = x[:, :, pad:-pad, pad:-pad] # 패딩 제거
        h -= 2 * pad
        w -= 2 * pad

    out = torch.zeros((B, C, h*f, w*f), device=x.device) # 원래 해상도로 복원해야 하기 때문에 h*f, w*f 크기의 텐서 생성

    idx = 0
    for i in range(f):
        for j in range(f):
            out[:, :, i::f, j::f] = x[idx*B:(idx+1)*B] # 배치에 쌓여있던 sub를 다시 원래 (i, j)위치에 복원하는 과정
            idx += 1

    return out


# ======================================================================================
# 4. DBSNl (공식 Blind-Spot Network)
# ======================================================================================

class CentralMaskedConv2d(nn.Conv2d):
    """
    커널 중앙 weight를 0으로 만들어
    identity mapping을 구조적으로 차단
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Conv2d 초기화 그대로 사용
        self.register_buffer("mask", torch.ones_like(self.weight)) # mask는 학습 파라미터가 아니라 버퍼로 등록하여 학습시키지는 않지만 저장 및 이동은 가능
        _, _, kH, kW = self.weight.shape 
        self.mask[:, :, kH//2, kW//2] = 0 # 커널 중앙 weight를 0으로 만들기 위한 마스크 생성 -> Idenitty mapping 방지

    def forward(self, x):
        self.weight.data = self.weight.data * self.mask # masked_weight = self.weight * self.mask 해서 conv에 넣는 형태가 더 안전함.
        return super().forward(x)


class DCl(nn.Module):
    """
    Dilated residual block
    - dilation으로 RF 확장
    - residual로 안정성 확보
    """

    def __init__(self, stride, ch):
        super(DCl, self).__init__()
        self.body = nn.Sequential(
            # dilation = stride  -> 커널 간 간격을 벌려 RF를 크게 확장
            nn.Conv2d(ch, ch, 3, padding=stride, dilation=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1) # 1x1 Conv로 채널로 mixing
        )

    def forward(self, x):
        return x + self.body(x) # residual connection


class DC_branchl(nn.Module):
    """
    Blind-Spot branch (stride=2 or 3)
    """

    def __init__(self, stride, ch, num_module):
        super(DC_branchl, self).__init__()

        layers = [
            # 중앙을 마스킹한 conv로 blind-spot 성질 유지
            CentralMaskedConv2d(ch, ch, kernel_size=2*stride-1, padding=stride-1), # stride = 2이면 kernel_size = 3, stride = 3이면 kernel_size = 5
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.ReLU(inplace=True),
        ]

        layers += [DCl(stride, ch) for _ in range(num_module)] # dilation = stride인 residual block을 num_module개 반복 -> 큰 RF 확보
        layers += [nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True)] # 마지막 채널 정리

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class DBSNl(nn.Module):
    """
    ✔ 공식 AP-BSN에서 사용하는 Blind-Spot Network
    ✔ 두 개의 dilation branch (2,3)
    """

    def __init__(self, in_ch=3, base_ch=128, num_module=9):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 1),
            nn.ReLU(inplace=True)
        )

        self.b1 = DC_branchl(2, base_ch, num_module)
        self.b2 = DC_branchl(3, base_ch, num_module)

        self.tail = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 1), # 두 개의 branch 결과를 채널 방향으로 concat 했으므로 base_ch * 2로 받고 다시 base_ch로 줄임
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, base_ch//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, in_ch, 1)
        )

    def forward(self, x):
        x = self.head(x)
        x = torch.cat([self.b1(x), self.b2(x)], dim=1) # 두 개의 branch 결과를 channel 방향으로 concat
        return self.tail(x)


# ======================================================================================
# 5. APBSN Wrapper + R³
# ======================================================================================

class APBSN(nn.Module):
    """
    AP-BSN 전체 파이프라인 + R³
    """

    def __init__(self):
        super().__init__()

        self.pd_a = 5 # Asymmetric PD, train 시 stride 5
        self.pd_b = 2 # Asymmetric PD, inference 시 stride 2
        self.pd_pad = 2 # PD에서 reflect padding 크기

        self.R3_T = 8 # R³ 반복 횟수
        self.R3_p = 0.16 # R³ 교체 확률

        self.bsn = DBSNl() # Blind-Spot Network

    def forward(self, x, pd=None):
        if pd is None:
            pd = self.pd_a # 기본적으로 train stride 5 사용

        if pd > 1:
            x = pd_down(x, pd, self.pd_pad)
            x = self.bsn(x)
            x = pd_up(x, pd, self.pd_pad)

        else: # PD 없이(pd = 1) 사용할 때, 그냥 BSN만 적용할 때에는 경계 아티팩트를 줄이기 위해서 reflect pad 후 conv하고 crop으로 원복
            x = F.pad(x, (2,2,2,2), mode="reflect")
            x = self.bsn(x)[:, :, 2:-2, 2:-2]

        return x

    @torch.no_grad()
    def denoise(self, x): # inference + R³
        """
        inference + Random Replacing Refinement
        """
        B, C, H, W = x.shape

        # H,W가 pd_b로 나누어 떨어지도록 최소 padding 크기를 계산
        # 예: pd_b=2인데 H가 홀수면 pad_h=1
        
        pad_h = (self.pd_b - H % self.pd_b) % self.pd_b
        pad_w = (self.pd_b - W % self.pd_b) % self.pd_b

        x = F.pad(x, (0,pad_w,0,pad_h), mode = 'reflect') # 우측, 하단에 패딩 적용

        base = self.forward(x, pd=self.pd_b) # PD stride = 2로 denoise 기본 결과
        outs = [] # R³ 결과들을 모아 평균 낼 리스트

        for _ in range(self.R3_T):
            mask = (torch.rand_like(x[:, :1]) < self.R3_p) # 해당 샘플링을 베르누이 샘플링이라고 함
            # 마스크 공간을 (H,W)로 만들고 그 마스크를 RGB 전체 채널에 동일하게 적용
            # 0~1 사이의 균등 난수 생성 후 p 확률보다 작으면 True(1), 아니면 False(0)
            # x[:, :1]는 (B, 1, H, W) 형태로 채널 1짜리 마스크 생성, 각 픽셀을 p 확률로 치환 대상으로 만드는 샘플

            tmp = base.clone() # base를 복사해서 일부 픽셀만 원본 noisy로 다시 치환하기 위해서 복사해둠


            # mask: [B, 1, H, W]
            # tmp: [B, C, H, W]
            # expand_as(target): target과 동일한 shape로 확장, 즉, [B, 1, H, W] -> [B, C, H, W]로 확장
            # mask가 True인 위치들에 대해서 tmp 값을 x 값으로 바꿔라 즉, 선택된 픽셀만 원본 noisy로 되돌린다.
            # mask가 False인 경우 base 값을 유지, mask가 Ture인 경우, x 값으로 치환
            tmp[mask.expand_as(tmp)] = x[mask.expand_as(x)]
            # mask를 [B, C, H, W] 형태로 확장해서, 선택된 위치는 원본 노이즈 픽셀로 교체
            # 의미: denoised 이미지에 "랜덤하게 noisy를 다시 주입"
            # 목적: 남아있는 상관 구조를 기대값 레벨에서 p배로 약화시키는 decorrelation


            tmp = F.pad(tmp, (2,2,2,2), mode="reflect") # 치환된 tmp에 대해서 경계 아티팩트를 줄이기 위해서 (left, right, top, bottom)으로 reflect padding 적용
            # pad를 2로 수행했으니까 tmp의 H, W가 각각 4씩 늘어남 -> 원래 크기인 H, W로 복원하기 위해서 패딩으로 늘어난 테두리 2픽셀씩을 잘라냄
            # 2:-2 -> 두 번째 픽셀부터 끝에서 두 번째 픽셀까지 선택
            tmp = self.bsn(tmp)[:, :, 2:-2, 2:-2] # edge articact를 제거하기 위해 패딩 후 BSN 통과시키고 다시 crop
            outs.append(tmp)

        out = torch.stack(outs).mean(0) # R³ 결과들을 쌓아서 평균 낸다, [T,B,C,H,W]로 쌓고 T 평균 => Monte Carlo 평균

        return out[:, :, :H, :W] # 처음 입력의 원래 H, W로 크기 복원


# ======================================================================================
# 6. SIDD Pair 수집 (NOISY, GT) + split(5:3:2)
# ======================================================================================

def collect_sidd_pairs(root):
    """
    SIDD_Small_sRGB_Only 구조에서 (noisy, gt) pair를 전부 수집

    네 예시:
    .../Data/0001_001_S6_00100_00060_3200_L/GT_SRGB_010.PNG
    .../Data/0001_001_S6_00100_00060_3200_L/NOISY_SRGB_010.PNG

    핵심:
    - 같은 폴더 안에서 NOISY_SRGB_XXX.PNG ↔ GT_SRGB_XXX.PNG가 1:1 대응
    """
    noisy_paths = glob.glob(os.path.join(root, "**/NOISY_SRGB_*.PNG"), recursive=True)
    noisy_paths = sorted(noisy_paths)

    pairs = []
    for npath in noisy_paths:
        # NOISY_SRGB_010.PNG -> GT_SRGB_010.PNG 로 바꿔서 gt 경로 생성
        gpath = npath.replace("NOISY_SRGB_", "GT_SRGB_")
        if os.path.exists(gpath):
            pairs.append((npath, gpath))
        else:
            # 혹시 확장자/대소문자 이슈가 있으면 여기서 걸림
            print(f"[WARN] GT not found for: {npath}")

    if len(pairs) == 0:
        raise RuntimeError("No (NOISY, GT) pairs found. Check root path.")

    return pairs


def split_pairs_5_3_2(pairs, seed=0):
    """
    pair 리스트를 5:3:2 비율로 train/val/test로 분할
    - 5:3:2 = 50% : 30% : 20%
    - seed로 재현성 유지
    """
    rng = random.Random(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)

    N = len(pairs)
    n_train = int(N * 0.5)
    n_val = int(N * 0.3)
    n_test = N - n_train - n_val  # 나머지는 test로 (합 보장)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"[Split] total={N}, train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


# ======================================================================================
# 7. Dataset 2종
#    (1) Train: noisy-only patch sampling (SSL)
#    (2) Val/Test: full-image pair (noisy, gt) for metric eval
# ======================================================================================

class SIDDTrainNoisyPatchDataset(Dataset):
    """
    ✅ 학습용: noisy-only SSL patch dataset
    - train_pairs에서 noisy 경로만 사용
    - 매 iteration 랜덤 이미지 선택 → 120x120 랜덤 크롭 → augmentation
    - __len__은 patches_per_epoch로 '가상' 길이
    """

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
        # 1) noisy 이미지 랜덤 선택
        img = Image.open(random.choice(self.noisy_paths)).convert("RGB")

        # 2) PIL -> torch float 텐서 [3,H,W], [0,1]
        img = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0

        _, H, W = img.shape
        ps = self.patch_size

        # (SIDD small은 충분히 크지만, 혹시 작은 이미지가 섞이면 방어)
        if H < ps or W < ps:
            # 작은 이미지면 center-crop 비슷하게 처리(최대한 안전하게)
            top = max(0, (H - ps) // 2)
            left = max(0, (W - ps) // 2)
            img = F.interpolate(img.unsqueeze(0), size=(max(H, ps), max(W, ps)), mode="bilinear", align_corners=False).squeeze(0)
            _, H, W = img.shape
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)

        patch = img[:, top:top+ps, left:left+ps]

        # 3) augmentation: rot90
        k = random.randint(0, 3)
        patch = torch.rot90(patch, k, dims=(1, 2))

        # 4) flip
        if random.random() < 0.5:
            patch = torch.flip(patch, dims=(2,))
        if random.random() < 0.5:
            patch = torch.flip(patch, dims=(1,))

        return patch


class SIDDPairImageDataset(Dataset):
    """
    ✅ 검증/테스트용: (noisy, gt) full-image pair dataset
    - 전체 이미지를 불러오므로 batch_size는 보통 1 권장(메모리/속도 안정)
    """

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

        # (C,H,W) float32
        return noisy, gt, npath


# ======================================================================================
# 8. Metric: PSNR / SSIM (RGB) - 우리가 늘 쓰는 "안전한" 구현
# ======================================================================================

def psnr_torch(pred, target, eps=1e-10):
    """
    pred, target: [B,C,H,W] in [0,1]
    PSNR = 10 * log10(1 / MSE)
    """
    mse = torch.mean((pred - target) ** 2, dim=(1,2,3))  # [B]
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean().item()


def ssim_torch(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    간단 SSIM (RGB 채널 평균) 구현
    - pred/target: [B,C,H,W] in [0,1]
    - 논문급 완전 정교 구현이 아니라, val/test 경향 확인용으로 충분히 안정적
    """
    # 1) Gaussian window 만들기
    def gaussian_window(ws, sigma=1.5, device="cpu", dtype=torch.float32):
        coords = torch.arange(ws, device=device, dtype=dtype) - ws // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        w = g[:, None] * g[None, :]
        return w

    B, C, H, W = pred.shape
    device = pred.device
    dtype = pred.dtype

    w = gaussian_window(window_size, device=device, dtype=dtype)  # [ws,ws]
    w = w.view(1, 1, window_size, window_size)
    w = w.repeat(C, 1, 1, 1)  # depthwise conv용 [C,1,ws,ws]

    # 2) 동일 padding으로 mean/var 계산
    pad = window_size // 2
    mu1 = F.conv2d(pred, w, padding=pad, groups=C)
    mu2 = F.conv2d(target, w, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, w, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, w, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, w, padding=pad, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    # [B,C,H,W] -> 채널 평균 -> 배치 평균
    return ssim_map.mean(dim=(1,2,3)).mean().item()

# ======================================================================================
# 9. Train / Val / Test 루프
# ======================================================================================

@dataclass
class Config:
    sidd_root: str = "/Users/im-woojin/Documents/GitHub/wannbe/GIST/SIDD_Small_sRGB_Only/Data"

    seed: int = 0

    # train patch
    patch_size: int = 120
    patches_per_epoch: int = 20000

    # optimization
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0

    # log / ckpt
    save_dir: str = "./checkpoints_apbsn"
    save_every: int = 1  # epoch 단위 저장

    # eval
    eval_max_images: int = 50  # val/test에서 너무 오래 걸리면 제한 (None이면 전체)


def get_device():
    # MPS 우선(네 환경), 없으면 CUDA, 없으면 CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    n = 0

    for batch in loader:
        # batch: [B,C,ps,ps] (noisy patch)
        noisy = batch.to(device)

        # AP-BSN 학습은 기본 forward(pd=5)로 간다고 보면 됨
        pred = model(noisy, pd=model.pd_a)

        # SSL: target을 noisy 자기 자신으로 둠 (blind-spot + PD로 identity 방지)
        loss = F.l1_loss(pred, noisy)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()
        n += 1

    return running / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, max_images=None):
    """
    val/test: full-image로 denoise() 실행 후 PSNR/SSIM 측정
    """
    model.eval()

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    for noisy, gt, path in loader:
        noisy = noisy.to(device)  # [1,C,H,W]
        gt = gt.to(device)

        # inference는 denoise()가 R³까지 포함
        pred = model.denoise(noisy)

        # clamp: 혹시 네트워크 출력이 미세하게 범위 벗어나면 metric 왜곡 방지
        pred = pred.clamp(0.0, 1.0)

        psnr_sum += psnr_torch(pred, gt)
        ssim_sum += ssim_torch(pred, gt)
        count += 1

        if (max_images is not None) and (count >= max_images):
            break

    if count == 0:
        return 0.0, 0.0

    return psnr_sum / count, ssim_sum / count


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
# 10. main: split -> train -> val -> best 저장 -> test
# ======================================================================================

def main():
    cfg = Config()

    # 재현성
    set_seed(cfg.seed)

    # pair 수집 + split
    pairs = collect_sidd_pairs(cfg.sidd_root)
    train_pairs, val_pairs, test_pairs = split_pairs_5_3_2(pairs, seed=cfg.seed)

    # dataset / loader
    train_ds = SIDDTrainNoisyPatchDataset(train_pairs, patch_size=cfg.patch_size, patches_per_epoch=cfg.patches_per_epoch)
    val_ds = SIDDPairImageDataset(val_pairs)
    test_ds = SIDDPairImageDataset(test_pairs)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,   # mac/mps에서는 0이 안정적일 때가 많음
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,    # full image는 1 권장
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # model / opt
    device = get_device()
    print(f"[Device] {device}")

    model = APBSN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.save_dir, exist_ok=True)

    best_psnr = -1.0

    # -------------------------
    # Train loop
    # -------------------------
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # val metric
        val_psnr, val_ssim = evaluate(model, val_loader, device, max_images=cfg.eval_max_images)

        print(f"[Epoch {epoch:03d}] train_L1={train_loss:.6f} | val_PSNR={val_psnr:.3f} | val_SSIM={val_ssim:.4f}")

        # checkpoint 저장
        if epoch % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(save_path, model, optimizer, epoch, best_psnr)

        # best 갱신
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = os.path.join(cfg.save_dir, "best.pt")
            save_checkpoint(best_path, model, optimizer, epoch, best_psnr)
            print(f"  -> [BEST] updated! best_PSNR={best_psnr:.3f} saved to {best_path}")

    # -------------------------
    # Test (best 로드 후 평가)
    # -------------------------
    best_path = os.path.join(cfg.save_dir, "best.pt")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, optimizer=None, map_location=device)
        print(f"[Load] best checkpoint loaded: {best_path}")
    else:
        print(f"[WARN] best checkpoint not found. Use last model.")

    test_psnr, test_ssim = evaluate(model, test_loader, device, max_images=cfg.eval_max_images)
    print(f"[TEST] PSNR={test_psnr:.3f} | SSIM={test_ssim:.4f}")


if __name__ == "__main__":
    main()



