import os
import random
import argparse
import glob
import shutil
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


# ================================================================
# Utils
# ================================================================
# iamge.shape = (C, H, W)
# view() -> 같은 메모리를 다른 모양으로 보기
# reshape() -> 새로운 메모리 할당, 복사본 생성
# permute() -> 차원 순서 변경
def torch_normalize(image, patch_size):
    mean = image.mean(dim = [1, 2], keepdim = True)
    std = image.std(dim = [1, 2], keepdim = True)

    # 분산 하한(variance collapse 방지)
    # C-BSN 논문에서 말한 정규화에서 0으로 나누는 상황을 방지하기 위함
    adjustde_std = torch.maximum(std, torch.tensor(1.0 / (patch_size * patch_size), device = image.device))

    normalized = (image - mean) / adjustde_std

    return normalized

def torch_augment(image, seed):
    # rotate 90도 단위 회전 + 좌우 반전 랜덤하게 수행
    rot_time = seed % 4
    image = torch.rot90(image, k = rot_time, dims = [1, 2])

    do_flip = (seed // 4) % 2
    if do_flip:
        image = torch.flip(image, dims = (2,))

    return image

def get_paramsnum(model: nn.Module):
    #.numel() : 텐서의 모든 원소 수를 반환 -> 텐서를 모델의 파라미터로 넘기면 그 모델의 파라미터 수를 알 수 있음
    total_params = sum(p.numel() for p in model.parameters()) # 학습 가능한 모델 파라미터 수 계산

    print("=" * 20)
    print(f"Network parameters : {total_params}")
    print("=" * 20)

    return total_params

def cpy_code(checkpoint_dir:str): # 실험 당시 코드 스냅샷 보존, 재현성을 위한 필수 도구
    os.makedirs(checkpoint_dir, exist_ok=True)

    py_files = glob.glob('./*.py')
    for file in py_files:
        shutil.copy(file, os.path.join(checkpoint_dir, os.path.basename(file)))

def batch_PSNR_255(noisy:np.ndarray, ref : np.ndarray):
    """
    PSNR 계산 (픽셀 값 범위 0~255)
    noisy: (B, C, H, W) np.ndarray
    ref: (B, C, H, W) np.ndarray
    """
    psnr = 0.0
    B = noisy.shape[0]

    for i in range(B):
        noisy_i = np.round(255*noisy[i].astype(np.float32)) # 입력을 0~1로 정규화했으니 255를 곱해 원래 범위로 복원(8-bit 이미지 스케일)
        ref_i = np.round(255*ref[i].astype(np.float32)) # GT 이미지도 노이즈 이미지와 같이 동일하게 수행

        psnr += compute_psnr(ref_i, noisy_i, data_range = 255) # data_range를 255로 지정하여 PSNR 계산 -> 이미지가 가질 수 있는 최대 픽셀 값
    
    return psnr / B

def to_vaild_image(image : torch.Tensor): # TensorBoard / PNG 저장용

    img = torch.clamp(image, 0.0, 1.0)
    img = (img * 255).to(torch.uint8)
    return img

def normalize(img: np.ndarray):
    """
    img: (B, H, W, C)
    """
    mean = np.mean(img, axis = (1, 2), keepdims = True)
    std = np.std(img, axis = (1, 2), keepdims = True)

    H, W = img.shape[1], img.shape[2]
    std = np.maximum(std, 1.0 / (H * W))

    img_norm = (img - mean) / std

    return img_norm, mean, std

def im2unit8(img: np.ndarray):
    # numpy image to uint8 image
    # grayscale / RGB 모두 대응
    # squeeze()로 (H, W, 1) → (H, W)
    img = np.clip(img, 0.0, 1.0) # 0~1 범위로 클리핑 - 이미지의 픽셀 값이 허용된 범위를 벗어나면 그 값을 강제로 범위 안으로 잘라버리는 것
    img = (img * 255).astype(np.uint8)

    return img.squeeze() # 차원에서 크기가 1인 축을 제거, grayscale 이미지의 경우, (H, W, 1)-> (H, W)로 변환, RGB 이미지는 (H, W, 3)로 유지
    # 학습과 추론 시에는 float(0~1)로 두고 연산량을 줄이고, 저장 및 시각화는 0~255로 두어서 우리가 보는 이미지로 바꾸는 것


# ================================================================
# Random Subsampler (RS)
# ================================================================
class RandomSubsampler(nn.Module):
    def __init__(self, ds_factor = 2):
        super().__init__()
        self.s = ds_factor # downsampling stride factor

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.s
        assert H % s == 0 and W % s == 0 # 이미지를 정확히 나누기 위한 조건

        # (B, C, H, W)
        # → (B, C, H/s, s, W/s, s)
        # H축 -> (H/s)개의 큰 블록 x s개의 내부 픽셀
        # W축 -> (W/s)개의 큰 블록 x s개의 내부 픽셀
        x = x.view(B, C, H//s, s, W//s, s) # reshaping
        x = x.permute(0, 2, 4, 1, 3, 5) # B, Hs, Ws, C, s, s

        # Random indexes for subsampling, 각 block마다 내부 좌표 무작위 선택 -> 이 block에서는 어떤 픽셀을 선택할지 결정
        idx_h = torch.randint(0, s, (B, H // s, W // s), device=x.device)
        idx_w = torch.randint(0, s, (B, H // s, W // s), device = x.device)

        out = []
        for b in range(B):
            out.append(x[b, # b번째 배치
                         torch.arange(H//s)[:, None], # block row indexes, shape (H/s, 1)
                         torch.arange(W//s)[None, :], # block column indexes, shape (1, W/s)
                            :, # 모든 channels
                            idx_h[b], # 무작위로 선택된 내부 row indexes, shape (H/s, W/s)
                            idx_w[b]]) # 무작위로 선택된 내부 column indexes, shape (H/s, W/s)
            # 각 블록에서 딱 1픽셀씩 뽑음
        return torch.stack(out).permute(0, 3, 1, 2) # (B, C, H/s, W/s)


# ===============================================================
# Space2Batch / Batch2Space
# ===============================================================
def S2B(x, stride):
    if stride == 1: return x # stride가 1이면 변화 없음

    B, C, H, W = x.shape
    s = stride
    assert H % s == 0 and W % s == 0 # s x s grid로 정확히 분해 가능해야 함

    x = x.view(B, C, H//s, s, W//s, s) # (B, C, H/s : 블록 row, s : 블록 내부 row, W/s : 블록 col, s : 블록 내부 col)

    x = x.permute(0, 3, 5, 1, 2, 4)  # B, s : offset row, s : offset col, C, H/s, W/s, s x s개의 서로 다른 격자

    x = x.reshape(B * s * s, C, H // s, W // s)  # (B*r*r, C, H/s, W/s), s×s개의 서로 다른 offset 격자를 각각 하나의 batch로 만든다
    return x


def B2S(x, stride, B):
    if stride == 1 : return x # stride가 1이면 변화 없음

    s = stride
    Br2, C, H, W = x.shape
    assert Br2 == B * s * s # 입력 배치 크기가 예상과 일치하는지 확인, 즉, batch 크기가 s x s의 배수인지 확인

    # (B*r*r, C, H, W) -> (B, r, r, C, H, W)
    # S2B의 역연산 과정
    x = x.view(B, s, s, C, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2)  # B, C, H, s, W, s
    x = x.reshape(B, C, H * s, W * s)

    return x
# ===============================================================
# Network
# ===============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation):
        super(ConvBlock, self).__init__()

        # Padding = same을 수동으로 구현
        # same padding: 출력 크기가 입력 크기와 동일하도록 패딩을 설정
        # 일반 conv2ddptj padding = (kernel_size // 2)
        # dilation이 적용되면 effective kernel size가 커지므로 padding도 dilation만큼 늘려줘야 함. -> padding = (kernel_size // 2) * dilation
        padding = (kernel_size // 2) * dilation
        
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = 1,
                              padding = padding,
                              dilation = dilation)
        
        nn.init.kaiming_normal_(self.conv.weight, mode = 'fan_in', nonlinearity = 'relu') #tensorflow의 variance_scaling_initializer(2.0)와 유사
        nn.init.zeros_(self.conv.bias)

        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)

        return x

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, mask, activation):
        super(MaskedConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation

        padding = (kernel_size // 2) * dilation

        self.padding = padding

        # 학습되는 커널 가중치를 직접 만드는 방식: nn.conv2d를 사용하지 않고, nn.Parameter로 가중치와 편향을 정의하여 이를 F.conv2d에서 사용하는 방식
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size)) # (out_channels, in_channels, H, W)
        self.bias = nn.Parameter(torch.zeros(out_channels)) # conv 결과에 각 출력 채널마다 하나씩 더해짐


        self.register_buffer("mask", mask) # mask는 학습되면 안 되는 값이니까 Parameter가 아니라 buffer로 등록, .to(device)로 이동 가능


        nn.init.kaiming_normal_(self.weight, mode = 'fan_in', nonlinearity = 'relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        w = self.weight * self.mask

        out = F.conv2d(x, w, bias = self.bias, stride = 1, padding = self.padding, dilation = self.dilation)

        if self.activation is not None:
            out = self.activation(out)

        return out

def make_blind_mask(size): # size x size커널에서 중앙을 0으로 만든 mask 생성 함수
    mask = torch.ones(1, 1, size, size, dtype = torch.float32) # mask tensor를 1로 채움, shape : (1, 1, size, size) 이는 weight과 곱해질 때 자동으로 브로드캐스팅 되도록 함
    mask[:, :, size // 2, size // 2] = 0 # 중앙 위치를 0으로 설정하여 마스킹 수행

    return mask

def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class DCM(nn.Module):
    def __init__(self, channels, dilation):
        super(DCM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size=3, stride = 1,
                               padding = (3 // 2) * dilation, dilation = dilation, bias = True)
        
        self.conv2 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 1, stride = 1, padding = 0, bias = True)

        nn.init.kaiming_normal_(self.conv1.weight, mode = 'fan_in', nonlinearity = 'relu')
        nn.init.zeros_(self.conv1.bias)

        nn.init.kaiming_normal_(self.conv2.weight, mode = 'fan_in', nonlinearity = 'relu')
        nn.init.zeros_(self.conv2.bias)

        self.activation = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = x

        f = self.activation(self.conv1(x))
        f = self.activation(self.conv2(f))

        return residual + f # skip-connection
    
class SIDDImageDataset(Dataset):
    def __init__(self, root_dir, patch_size = 240, return_gt = False):
        super(SIDDImageDataset, self).__init__()

        self.patch_size = patch_size
        self.return_gt = return_gt

        self.noisy_files = sorted(glob.glob(os.path.join(root_dir, '*', 'NOISY_SRGB_*.png')) +
        glob.glob(os.path.join(root_dir, '*', 'NOISY_SRGB_*.PNG')) # os.path.join(root_dir, '*', 'NOISY_SRGB_*.png') -> /root_dir/*/NOISY_SRGB_*.png, glob.glob()-> * 이것을 실제 파일 데이터셋 경로 리스트로 바꿔주는 함수
)

        assert len(self.noisy_files) > 0, "No noisy images found in the specified directory."

    def __len__(self):
        return len(self.noisy_files) 
    
    def __getitem__(self, idx):
        '''
        __getitem__의 process:
        1. 이미지 로드 (noisy, gt) -> PIL Image to Numpy
        2. 패치 추출 (무작위 위치) -> 좌상단 기준 정하고 패치 크기만큼 자르기
        3. 전처리: Tensor 변환, data augmentation, 정규화
        '''
        noisy_path = self.noisy_files[idx] # 노이즈 이미지 경로
        gt_path = noisy_path.replace('NOISY_SRGB', 'GT_SRGB') # 노이즈 이미지 경로에서 GT 이미지 경로로 변환

        noisy = Image.open(noisy_path).convert('RGB')
        noisy = np.array(noisy).astype(np.float32) / 255.0

        if self.return_gt:
            gt = Image.open(gt_path).convert('RGB')
            gt = np.array(gt).astype(np.float32) / 255.0

        H, W, _ = noisy.shape
        ps = self.patch_size
        # 패치 추출의 좌상단 좌표 무작위 선택
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        # 패치 추출
        noisy_patch = noisy[top:top + ps, left:left + ps, :]
        if self.return_gt:
            gt_patch = gt[top:top + ps, left: left + ps, :]
        
        # Pre-processing: Tensor 변환, data augmentation, 정규화
        seed = random.randint(0, 7)
        noisy_patch = torch.from_numpy(noisy_patch).permute(2, 0, 1) # (C, H, W)
        noisy_patch = torch_augment(noisy_patch, seed)
        noisy_patch = torch_normalize(noisy_patch, ps)

        if self.return_gt:
            gt_patch = torch.from_numpy(gt_patch).permute(2, 0, 1) # (C, H, W)
            return noisy_patch, gt_patch
        
        return noisy_patch
    
def get_dataloaders(args):
    # --------------------
    # Train dataset
    # --------------------
    train_dataset = SIDDImageDataset(
        root_dir=args.data_root,
        patch_size=args.patch_size,
        return_gt=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # --------------------
    # Validation dataset
    # --------------------
    val_dataset = SIDDImageDataset(
        root_dir=args.data_root,
        patch_size=args.val_patch_size,
        return_gt=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader





class CBSN(nn.Module):
    def __init__(self, in_channels, filters = 128, num_module = 9, is_masked = True):
        super(CBSN, self).__init__()


        # 마스크 생성
        if is_masked:
            mask3 = make_blind_mask(3)
            mask5 = make_blind_mask(5)

        else:
            mask3 = torch.ones(1, 1, 3, 3, dtype = torch.float32)
            mask5 = torch.ones(1, 1, 5, 5, dtype= torch.float32)

        # initial projection
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = filters, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU(inplace = True)
        )

        # Branch 1, (3x3 masked conv branch)
        self.branch1_conv1 = MaskedConv2d(in_channels = filters, out_channels = filters, kernel_size = 3, dilation = 1, mask = mask3, activation = nn.ReLU(inplace = True))
        
        self.branch1_conv2 = nn.Sequential(
            nn.Conv2d(in_channels = filters, out_channels = filters, kernel_size = 1),
            nn.ReLU(inplace = True)
        )

        self.branch1_conv3 = nn.Sequential(
            nn.Conv2d(in_channels = filters, out_channels = filters, kernel_size = 1),
            nn.ReLU(inplace = True)
        )

        self.branch1_DCMs = nn.ModuleList([
            DCM(channels = filters, dilation = 2) for _ in range(num_module) # dilation 2를 사용하는데 이거 github에서 확인함, ModuleList로 여러 개 쌓기
        ])

        self.branch1_conv4 = nn.Sequential(
            nn.Conv2d(in_channels = filters, out_channels = filters, kernel_size = 1),
            nn.ReLU(inplace = True)
        )

        # Branch 2, (5x5 masked conv branch)
        self.branch2_conv1 = MaskedConv2d(in_channels = filters, out_channels = filters, kernel_size = 5, dilation = 1, mask = mask5, activation = nn.ReLU(inplace = True))

        self.branch2_conv2 = nn.Sequential(
            nn.Conv2d(in_channels = filters, out_channels = filters, kernel_size = 1),
            nn.ReLU(inplace = True)
        )

        self.branch2_conv3 = nn.Sequential(
            nn.Conv2d(in_channels = filters, out_channels = filters, kernel_size = 1),
            nn.ReLU(inplace = True)
        )
        
        self.branch2_DCMs = nn.ModuleList([
            DCM(channels = filters, dilation = 3) for _ in range(num_module)# dilation 3를 사용하는데 이거 github에서 확인함, ModuleList로 여러 개 쌓기
        ])

        self.branch2_conv4 = nn.Sequential(
            nn.Conv2d(in_channels = filters, out_channels = filters, kernel_size = 1),
            nn.ReLU(inplace = True)
        )

        # fusion convs
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels = filters * 2, out_channels = filters, kernel_size =1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = filters, out_channels=64, kernel_size =1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 1)
        )

    def forward(self, x):
        x = self.conv0(x)

        # Branch 1
        b1 = self.branch1_conv1(x)
        b1 = self.branch1_conv2(b1)
        b1 = self.branch1_conv3(b1)
        for dcm in self.branch1_DCMs:
            b1 = dcm(b1)
        b1 = self.branch1_conv4(b1)

        # Branch 2
        b2 = self.branch2_conv1(x)
        b2 = self.branch2_conv2(b2)
        b2 = self.branch2_conv3(b2)
        for dcm in self.branch2_DCMs:
            b2 = dcm(b2)
        b2 = self.branch2_conv4(b2)

        # Concatenate

        concat = torch.cat([b1, b2], dim = 1) # channel dim 기준으로 concat
        out = self.fusion(concat)

        return out
    

# ================================================================
model = CBSN(in_channels = 3, filters = 128, num_module = 9, is_masked = True)
model.apply(init_weights)


# ================================================================
# Validation
# ================================================================
def validate(model, val_loader, device, writer=None, step=None):
    model.eval()
    psnr_total = 0.0
    count = 0

    with torch.no_grad():
        for i, (noisy, gt) in enumerate(val_loader):
            noisy = noisy.to(device)
            gt = gt.to(device)

            pred = model(noisy)
            pred = torch.clamp(pred, 0.0, 1.0)

            if i == 0 and writer is not None:
                writer.add_image(
                    "Val/Noisy",
                    to_vaild_image(noisy[0]),
                    step
                )
                writer.add_image(
                    "Val/Denoised",
                    to_vaild_image(pred[0]),
                    step
                )
                writer.add_image(
                    "Val/GT",
                    to_vaild_image(gt[0]),
                    step
                )

            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()
            psnr_total += batch_PSNR_255(pred_np, gt_np)
            count += 1

    return psnr_total / count


def inference(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # -------------------------
    # Model params
    # -------------------------
    N = args.num_block
    filters = args.channel
    pad = 32  # reflect padding size

    image_files = glob.glob(args.imagepath)
    savepath = args.savepath
    os.makedirs(savepath, exist_ok=True)

    # -------------------------
    # Model (unmasked CBSN)
    # -------------------------
    model = CBSN(
        in_channels=3,
        filters=filters,
        num_module=N,
        is_masked=False
    ).to(device)

    # -------------------------
    # Load checkpoint
    # -------------------------
    ckpt = torch.load(args.modelpath, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    print("✔ Restore model from:", args.modelpath)

    # -------------------------
    # Inference loop
    # -------------------------
    with torch.no_grad():
        for imgname in image_files:
            # 1. Read image
            img = Image.open(imgname).convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0  # (H, W, C)

            # 2. Expand batch dim
            img = img[None, ...]  # (1, H, W, C)

            # 3. Normalize (same as training)
            img_norm, mean, std = normalize(img)

            # 4. To torch tensor
            img_norm = torch.from_numpy(img_norm).permute(0, 3, 1, 2).to(device)

            # 5. Reflect padding
            img_norm = torch.nn.functional.pad(
                img_norm,
                pad=(pad, pad, pad, pad),
                mode="reflect"
            )

            # 6. Forward
            pred = model(img_norm)

            # 7. Remove padding
            pred = pred[:, :, pad:-pad, pad:-pad]

            # 8. Back to numpy
            pred = pred.permute(0, 2, 3, 1).cpu().numpy()

            # 9. Denormalize
            pred = std * pred + mean

            # 10. uint8 변환
            pred = im2unit8(pred)

            # 11. Save
            save_name = os.path.join(savepath, os.path.basename(imgname))
            Image.fromarray(pred).save(save_name)

            print("Saved:", save_name)

# ================================================================
# Trainer
# ================================================================

def trainer(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )

    # --------------------
    # Dataloader
    # --------------------
    train_loader, val_loader = get_dataloaders(args)
    train_iter = iter(train_loader)

    # --------------------
    # Model
    # --------------------
    model = CBSN(
        in_channels=3,
        filters=128,
        num_module=9,
        is_masked=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100000, gamma=0.5
    )

    writer = SummaryWriter(log_dir=f'./checkpoints/{args.name}')

    rs = RandomSubsampler(ds_factor=args.stride_i).to(device)

    global_step = 0

    # --------------------
    # Training loop
    # --------------------
    for step in range(args.max_iter):
        model.train()

        try:
            image_train = next(train_iter).to(device)
        except StopIteration:
            train_iter = iter(train_loader)
            image_train = next(train_iter).to(device)

        # ---------- Downsampling ----------
        image_ds1 = S2B(image_train, args.stride_b)
        image_ds2 = rs(image_train)

        # ---------- Forward ----------
        out_masked1 = model(image_ds1)
        out_masked1 = B2S(
            out_masked1, args.stride_b, image_train.shape[0]
        )

        out_masked2 = model(image_ds2)
        out_unmasked = model(image_train)
        out_unmasked_ds = rs(out_unmasked)

        # ---------- Loss ----------
        blind_loss = torch.mean(torch.abs(out_masked1 - image_train))
        self_loss = torch.mean(torch.abs(out_unmasked - image_train))
        invariance_loss = torch.mean(
            torch.abs(out_unmasked_ds - out_masked2.detach())
        )

        schedule = min(global_step / 200000.0, 1.0)

        total_loss = (
            blind_loss
            + schedule * (self_loss + args.lambda_inv * invariance_loss)
        )

        # ---------- Backprop ----------
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # ---------- Logging ----------
        if step % args.print_every == 0:
            print(
                f"[{step}] total={total_loss.item():.4f} | "
                f"blind={blind_loss.item():.4f} | "
                f"self={self_loss.item():.4f} | "
                f"inv={invariance_loss.item():.4f}"
            )

            writer.add_scalar("Train/total_loss", total_loss, step)
            writer.add_scalar("Train/blind_loss", blind_loss, step)
            writer.add_scalar("Train/self_loss", self_loss, step)
            writer.add_scalar("Train/invariance_loss", invariance_loss, step)

            # noisy / pred 비교 (첫 배치 1장만)
            noisy_vis = to_vaild_image(image_train[0])
            pred_vis = to_vaild_image(out_unmasked[0])
        
            writer.add_image(
                "Train/Noisy",
                noisy_vis,
                step
            )
        
            writer.add_image(
                "Train/Denoised",
                pred_vis,
                step
            )

        # ---------- Validation ----------
        if step % args.val_every == 0:
            psnr = validate(model, val_loader, device)
            print(f"[Val @ {step}] PSNR = {psnr:.2f}")
            writer.add_scalar("Val/PSNR", psnr, step)

        # ---------- Checkpoint ----------
        if step % args.save_every == 0:
            torch.save(
                model.state_dict(),
                f'./checkpoints/{args.name}/CBSN_iter{step}.pth'
            )

        global_step += 1

# ================================================================
# main
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBSN PyTorch")

    # --------------------
    # Mode
    # --------------------
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "inference", "validate"],
                        help="Execution mode")

    # --------------------
    # Training params
    # --------------------
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=400000)
    parser.add_argument("--lr", type=float, default=1e-4)

    # --------------------
    # Model params
    # --------------------
    parser.add_argument("--channel", type=int, default=128)
    parser.add_argument("--num_block", type=int, default=9)
    parser.add_argument("--lambda_inv", type=float, default=2.0)

    # --------------------
    # CBSN strides
    # --------------------
    parser.add_argument("--stride_b", type=int, default=5)
    parser.add_argument("--stride_i", type=int, default=2)

    # --------------------
    # Dataset
    # --------------------
    parser.add_argument("--data_root", type=str,
                        default="/home/work/AP-BSN/SIDD_Small_sRGB_Only/Data")
    parser.add_argument("--patch_size", type=int, default=240)
    parser.add_argument("--val_patch_size", type=int, default=256)

    # --------------------
    # Logging / checkpoint
    # --------------------
    parser.add_argument("--name", type=str, default="CBSN_SIDD")
    parser.add_argument("--print_every", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=10000)

    # --------------------
    # Inference options
    # --------------------
    parser.add_argument("--imagepath", type=str,
                        help="Inference image glob path")
    parser.add_argument("--savepath", type=str,
                        default="./results")
    parser.add_argument("--modelpath", type=str,
                        help="Checkpoint path for inference / validation")

    # --------------------
    # Device
    # --------------------
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])

    args = parser.parse_args()

    # --------------------
    # Device resolution
    # --------------------
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")

    elif args.device == "mps":
        device = torch.device("mps")

    elif args.device == "cpu":
        device = torch.device("cpu")

    else:  # auto
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Device: {device}")

    # ============================================================
    # Execute
    # ============================================================
    if args.mode == "train":
        trainer(args)

    elif args.mode == "inference":
        assert args.imagepath is not None
        assert args.modelpath is not None
        inference(args)

    elif args.mode == "validate":
        assert args.modelpath is not None

        _, val_loader = get_dataloaders(args)
        model = CBSN(
            in_channels=3,
            filters=args.channel,
            num_module=args.num_block,
            is_masked=False
        ).to(device)

        model.load_state_dict(
            torch.load(args.modelpath, map_location=device)
        )

        psnr = validate(model, val_loader, device)
        print(f"[VALIDATION] PSNR = {psnr:.2f}")

# python C_BSN_self.py --mode train --device cuda --gpu 0

# python train.py \
#   --mode inference \
#   --imagepath "test_images/*.png" \
#   --modelpath checkpoints/CBSN_SIDD/CBSN_iter500000.pth \
#   --savepath results \
#   --device cuda


# python train.py \
#   --mode validate \
#   --modelpath checkpoints/CBSN_SIDD/CBSN_iter500000.pth
