# ============================================================
# Noise2Void (N2V) - "논문 스타일" steps 기반 학습 버전 (PyTorch)
#
# ✅ 목표(논문 핵심 아이디어를 코드로 옮긴 것)
# 1) Clean GT 없이(noisy only) 학습한다.
# 2) 입력 패치에서 일부 픽셀(Blind-Spot)을 "가려서/치환해서" 그 위치 값을 주변 정보로만 예측하도록 만든다.
# 3) 손실(loss)은 오직 blind-spot 위치에서만 계산한다. (그 외 위치는 loss=0 취급)
# 4) epoch(데이터 몇 바퀴) 대신 steps(optimizer 업데이트 횟수)로 학습 길이를 제어한다.
#
# ✅ 전체 흐름(필수로 이해)
# [이미지 로드] -> [64x64 랜덤 크롭] -> [stratified로 mask 좌표 N개 선택]
# -> [그 좌표 픽셀을 주변 값으로 치환(blind-spot)] -> [모델 출력]
# -> [mask 좌표에서만 MSE 계산] -> [optimizer step]
# ============================================================

import os                                         # 파일/폴더 경로 조작(예: join, isdir, makedirs)에 필요
import math                                       # sqrt, log 같은 수학 함수(예: stratified sampling 격자 계산)에 필요
import random                                     # 랜덤 crop, 랜덤 mask 좌표, 랜덤 치환 픽셀 선택 등에 필요
from glob import glob                             # 폴더에서 특정 확장자 파일들을 찾는(패턴 매칭) 데 필요
from typing import List, Tuple, Optional           # 타입 힌트(가독성/IDE 도움)용

import numpy as np                                # 이미지 배열 변환, PSNR 계산 등 수치 연산에 필요
from PIL import Image                             # 이미지 파일 로드/저장(PNG/JPG 등)용

import torch                                      # PyTorch 핵심 텐서/디바이스/저장 로드
import torch.nn as nn                             # nn.Module, Conv2d 등 네트워크 레이어 구성
import torch.nn.functional as F                   # pad, rot90 같은 함수형 연산(레이어 없이) 사용
from torch.utils.data import Dataset, DataLoader  # 커스텀 데이터셋 + 배치 로딩(미니배치 학습)


# ---------------------------
# 0) Device
# ---------------------------
# ✅ 학습/추론을 어떤 디바이스에서 돌릴지 자동 선택
# - Apple Silicon이면 MPS
# - 그 외 CUDA 가능하면 CUDA
# - 둘 다 아니면 CPU
device = torch.device(                            # torch.device 객체를 만들어 모델/텐서를 올릴 장치 결정
    "mps" if torch.backends.mps.is_available()     # (조건1) MPS 사용 가능? -> "mps" 선택 (Mac Apple Silicon)
    else "cuda" if torch.cuda.is_available()       # (조건2) CUDA 사용 가능? -> "cuda" 선택 (NVIDIA GPU)
    else "cpu"                                    # (조건3) 둘 다 아니면 -> CPU
)


# ---------------------------
# 1) Image I/O
# ---------------------------
# ✅ 데이터셋 폴더에서 "이미지 파일"만 찾기 위한 확장자 목록
# - tuple로 둔 이유: 불변(immutable)이라 실수로 바뀌는 걸 방지하기 좋음
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")  # 검색할 확장자들(추가/삭제 가능)

def list_images(root: str) -> List[str]:          # root: 이미지들이 들어있는 폴더 경로(str)
    """
    ✅ 역할:
    - root 폴더 아래에 있는 모든 이미지 파일 경로를 재귀적으로 수집해서 리스트로 반환

    ✅ args:
    - root (str): 최상위 폴더 경로 (예: "datasets/BSD400")

    ✅ return:
    - List[str]: 이미지 파일들의 전체 경로 리스트
    """
    files = []                                    # 이미지 파일 경로들을 담을 리스트
    for ext in IMG_EXTENSIONS:                    # 정의한 확장자들(.png, .jpg, ...)에 대해 반복
        # glob을 사용해서 root/**/ *{ext} 패턴을 전부 찾음 (recursive=True로 하위 폴더 포함)
        files.extend(glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(files)                          # 순서 고정(재현성/디버깅 편의) 위해 정렬해서 반환

def load_image(path: str, grayscale: bool = True) -> torch.Tensor:  # path: 이미지 파일 경로, grayscale: 흑백 로드 여부
    """
    ✅ 역할:
    - 이미지 파일을 읽어서 PyTorch 텐서로 변환
    - pixel 값을 [0, 255] -> [0, 1]로 정규화
    - 텐서 shape을 (C, H, W)로 맞춤

    ✅ args:
    - path (str): 이미지 파일 경로
    - grayscale (bool): True면 흑백(1채널), False면 RGB(3채널)

    ✅ return:
    - torch.Tensor: shape [C,H,W], dtype float32, 값 범위 [0,1]
    """
    img = Image.open(path)                        # PIL로 이미지 파일 읽기(메모리에 로드)
    img = img.convert("L") if grayscale else img.convert("RGB")  # L=grayscale(1채널), RGB=3채널로 변환

    arr = np.array(img, dtype=np.float32) / 255.0 # PIL 이미지 -> numpy 배열(float32)로 변환 후 정규화([0,1])

    if arr.ndim == 2:                             # grayscale면 arr shape이 [H,W] (2차원)
        arr = arr[None, ...]                      # 채널 차원 추가 -> [1,H,W] (PyTorch는 채널 먼저)
    else:                                         # RGB면 arr shape이 [H,W,3]
        arr = arr.transpose(2, 0, 1)              # [H,W,C] -> [C,H,W]로 축 순서 변경

    return torch.from_numpy(arr)                  # numpy -> torch Tensor로 변환(복사 없이 view 가능)


# ---------------------------
# 2) Stratified sampling for N points
# ---------------------------
def stratified_sample_coords(patch_h: int, patch_w: int, n_points: int) -> List[Tuple[int, int]]:

    # ------------------------------------------------------------
# Stratified (Hierarchical) Sampling 설명 주석
# ------------------------------------------------------------
# 목적:
# - 64x64 patch 내부에서 n_points(예: 64)개의 픽셀 좌표를 선택
# - 좌표들이 한 곳에 몰리지 않도록 patch 전체에 고르게 분포시키기 위함
#
# ------------------------------------------------------------
# g의 의미
# ------------------------------------------------------------
# g는 patch를 "몇 x 몇" 격자로 나눌지를 정하는 값
# n_points = 64라면:
#   sqrt(64) = 8 → g = 8
# 즉,
#   patch를 8 x 8 = 64개의 구역(cell)으로 나눈다
#
# 이렇게 하면:
# - 각 구역(cell)에서 좌표를 1개씩 선택 가능
# - 결과적으로 총 64개의 좌표가 patch 전체에 고르게 퍼진다
#
# g*g < n_points 인 경우(g=7이면 49칸):
# - g를 1 증가시켜(g=8) 충분한 cell 수를 확보
#
# ------------------------------------------------------------
# cell의 의미
# ------------------------------------------------------------
# cell은 patch를 g x g로 나눴을 때 생기는 "각 작은 구역(블록)"
#
# 예:
#   patch_h = 64, patch_w = 64, g = 8
#   cell_h = 64 / 8 = 8
#   cell_w = 64 / 8 = 8
#
# 즉,
#   64x64 patch를
#   8x8 크기의 cell 64개로 쪼갠다
#
# 각 cell의 역할:
# - "이 구역 안에서는 blind-spot 픽셀을 최소 1개 선택하자"
# - masking 픽셀이 공간적으로 고르게 분포되도록 강제
#
# ------------------------------------------------------------
# 계층적(hierarchical)이라는 의미
# ------------------------------------------------------------
# 1단계(전역, coarse):
# - patch 전체를 g x g cell로 분할
# - blind-spot의 전역적 분포를 균등하게 제어
#
# 2단계(지역, fine):
# - 각 cell 내부에서 좌표를 랜덤으로 1개 선택
# - 랜덤성을 유지하면서도 clustering 방지
#
# ------------------------------------------------------------
# 결과
# ------------------------------------------------------------
# - 각 cell에서 1개씩 좌표 선택 → 총 n_points개
# - blind-spot 픽셀이 patch 전체 공간에 고르게 퍼짐
# - N2V에서 loss가 계산되는 위치가 특정 영역에 몰리지 않음
# - receptive field 겹침/편향 감소, 학습 안정성 향상

    """
    ✅ 역할:
    - patch 내부에서 마스킹할 좌표 N개를 고른다.
    - "한 곳에 몰리지 않도록" patch를 격자로 나눠 고르게 뽑는 Stratified Sampling.

    ✅ args:
    - patch_h (int): 패치 높이(예: 64)
    - patch_w (int): 패치 너비(예: 64)
    - n_points (int): 뽑을 좌표 개수(예: 64)

    ✅ return:
    - List[(y,x)]: y는 [0,patch_h-1], x는 [0,patch_w-1] 범위의 좌표들

    ✅ 왜 stratified?
    - 그냥 랜덤 샘플링하면 mask가 특정 구역에 몰려 학습 신호가 불균형해질 수 있음
    - patch 전체에 mask가 퍼지면 더 안정적으로 주변 문맥(context)을 학습하기 좋음
    """
    g = int(math.sqrt(n_points))                  # n_points의 제곱근을 정수로 -> 격자 한 변 크기 후보
    g = max(g, 1)                                 # g가 0이 되지 않도록 최소 1
    if g * g < n_points:                          # g*g가 n_points를 못 채우면
        g += 1                                    # 한 단계 키워서 충분히 많은 셀을 만들기

    cell_h = patch_h / g                          # 한 셀의 높이(실수)
    cell_w = patch_w / g                          # 한 셀의 너비(실수)

    coords = []                                   # 결과 좌표 리스트
    for gy in range(g):                           # 격자 y 인덱스(0..g-1)
        for gx in range(g):                       # 격자 x 인덱스(0..g-1)
            y0 = int(round(gy * cell_h))          # 현재 셀의 y 시작(정수)
            y1 = int(round((gy + 1) * cell_h))    # 현재 셀의 y 끝(정수)
            x0 = int(round(gx * cell_w))          # 현재 셀의 x 시작(정수)
            x1 = int(round((gx + 1) * cell_w))    # 현재 셀의 x 끝(정수)

            y1 = min(y1, patch_h)                 # 셀 경계가 patch 밖으로 나가지 않게 clamp
            x1 = min(x1, patch_w)

            if y0 >= y1 or x0 >= x1:              # 셀이 너무 작아서 유효 범위가 없으면
                continue                          # 해당 셀은 건너뜀

            yy = random.randint(y0, y1 - 1)       # 셀 내부 y를 균등 랜덤으로 1개 선택
            xx = random.randint(x0, x1 - 1)       # 셀 내부 x를 균등 랜덤으로 1개 선택
            coords.append((yy, xx))               # (y,x) 좌표를 리스트에 추가

    random.shuffle(coords)                        # 좌표 순서 랜덤 섞기(앞쪽만 쓰면 bias가 생길 수 있어서)
    if len(coords) >= n_points:                   # 충분히 많이 뽑혔다면
        coords = coords[:n_points]                # 앞에서 n_points개만 사용
    else:                                         # 부족하다면
        need = n_points - len(coords)             # 부족한 개수 계산
        for _ in range(need):                     # 부족한 개수만큼
            coords.append(                         # patch 전체에서 완전 랜덤으로 좌표 추가
                (random.randint(0, patch_h - 1), random.randint(0, patch_w - 1))
            )

    return coords                                 # 최종 좌표 리스트 반환


# ---------------------------
# 3) N2V masking (blind-spot replacement)
# ---------------------------
def apply_n2v_masking(
    patch: torch.Tensor,                          # patch: [C,H,W] float, 값 범위 [0,1]
    coords: List[Tuple[int, int]],                # coords: blind-spot 좌표들 [(y,x), ...]
    radius: int = 5                               # radius: 주변에서 대체 픽셀을 찾을 범위(정수)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ✅ 역할:
    - coords 위치를 blind-spot으로 만들기 위해 해당 픽셀 값을 "주변 픽셀 값"으로 치환한다.
    - 이렇게 해야 모델이 (y,x)의 원래 값을 입력에서 그대로 보고 복사하는 shortcut을 못 한다.

    ✅ args:
    - patch (Tensor): [C,H,W] 패치 (noisy)
    - coords (List[Tuple[int,int]]): 가릴 좌표들
    - radius (int): 치환에 사용할 주변 후보 범위 (예: 5면 11x11 window 안에서 선택)

    ✅ return:
    - masked_patch (Tensor): [C,H,W] 입력으로 사용할 패치(일부 픽셀 값이 치환됨)
    - mask (Tensor): [1,H,W] loss를 계산할 위치만 1인 마스크
    """
    C, H, W = patch.shape                         # 채널/높이/너비 추출
    masked = patch.clone()                        # 원본 patch는 target으로도 쓰니까 복사해서 수정(원본 보존)
    mask = torch.zeros((1, H, W), dtype=torch.float32)  # loss 위치 표시용 마스크 초기화(전부 0)

    for (y, x) in coords:                         # 마스킹할 좌표 하나씩 처리
        mask[0, y, x] = 1.0                       # 이 위치는 loss 계산 대상이라고 1로 표시

        # 주변 픽셀로 치환 시도: 최대 10번 랜덤 샘플링해서 유효한 이웃을 찾음
        for _ in range(10):                       # 10번 시도(너무 오래 걸리지 않게 상한)
            dy = random.randint(-radius, radius)  # y 방향 오프셋(예: -5~+5)
            dx = random.randint(-radius, radius)  # x 방향 오프셋(예: -5~+5)
            ny = y + dy                           # 후보 이웃 y
            nx = x + dx                           # 후보 이웃 x

            # 조건1: 이미지 안쪽 범위인지
            # 조건2: (dy,dx)=(0,0)이면 자기 자신이므로 제외 (정답 픽셀을 그대로 넣으면 의미 없음)
            if 0 <= ny < H and 0 <= nx < W and (dy != 0 or dx != 0):
                masked[:, y, x] = patch[:, ny, nx]  # (y,x)의 값을 이웃 픽셀 값으로 치환
                break                               # 성공했으면 반복 종료
        else:
            # 위 10번에서 실패한 경우(예: radius가 너무 작고 경계 문제 등) fallback 로직
            yy = random.randint(0, H - 1)           # patch 전체에서 랜덤 y
            xx = random.randint(0, W - 1)           # patch 전체에서 랜덤 x
            if yy == y and xx == x:                 # 혹시 자기 자신이면
                yy = (yy + 1) % H                   # y를 한 칸 이동해서 자기 자신 회피
            masked[:, y, x] = patch[:, yy, xx]      # 그 픽셀로 치환

    return masked, mask                              # masked 입력 + loss mask 반환


# ---------------------------
# 4) Dataset: random patch + masking
# ---------------------------
class N2VPatchDataset(Dataset):
    """
    ✅ 역할:
    - "이미지 파일"에서 매번 랜덤한 패치를 뽑고(masking 포함) 학습 샘플을 생성

    ✅ 출력 1개 샘플:
    - masked_patch: 모델 입력(일부 픽셀 치환된 패치)
    - target_patch: 정답(원본 noisy 패치)
    - mask: loss를 계산할 좌표만 1

    ✅ 왜 이렇게 만드는가?
    - N2V는 clean 정답이 없고, noisy patch 그 자체를 target으로 쓰되
      입력에서 target 픽셀을 못 보게(blind-spot) 만들어야만 의미가 있다.
    """
    def __init__(
        self,
        image_root_or_list,                        # (str or list) 이미지 폴더 경로 or 이미지 경로 리스트
        patch_size: int = 64,                      # patch 한 변 크기(기본 64)
        n_masked: int = 64,                        # patch 내부에서 mask할 좌표 수(기본 64)
        grayscale: bool = True,                    # 흑백 학습 여부(True면 1채널)
        neighbor_radius: int = 5,                  # 치환에 쓸 이웃 반경(radius)
        augment: bool = True,                      # flip/rot augmentation 적용 여부
    ):
        if isinstance(image_root_or_list, str):    # 입력이 폴더 경로(str)라면
            image_paths = list_images(image_root_or_list)  # 폴더에서 이미지 경로 리스트 생성
        else:                                      # 아니면 이미 리스트로 들어왔다고 보고
            image_paths = list(image_root_or_list) # 리스트로 복사(안전)

        if len(image_paths) == 0:                  # 이미지가 하나도 없으면
            raise RuntimeError("No images found. Check your dataset path/root.")  # 즉시 에러

        self.image_paths = image_paths             # 이미지 경로 리스트 저장
        self.patch_size = patch_size               # patch 크기 저장
        self.n_masked = n_masked                   # mask 포인트 수 저장
        self.grayscale = grayscale                 # grayscale 여부 저장
        self.neighbor_radius = neighbor_radius     # radius 저장
        self.augment = augment                     # augment 여부 저장

    def __len__(self):
        # steps 기반이라 길이가 큰 의미는 없지만 DataLoader가 epoch 구성에 사용하므로 이미지 수 반환
        return len(self.image_paths)

    def _random_crop(self, img: torch.Tensor) -> torch.Tensor:
        """
        ✅ 역할:
        - img에서 patch_size x patch_size 크기로 랜덤 crop

        ✅ 왜 random crop?
        - 전체 이미지를 통째로 넣으면 연산량이 크고, N2V는 patch 기반으로 학습하는 게 일반적
        - 다양한 지역에서 local statistics를 학습하려면 랜덤 crop이 유리
        """
        C, H, W = img.shape                        # 입력 이미지 크기 확인
        ps = self.patch_size                       # patch size

        if H < ps or W < ps:                       # 이미지가 patch보다 작으면
            pad_h = max(0, ps - H)                 # 필요한 padding 높이
            pad_w = max(0, ps - W)                 # 필요한 padding 너비
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")  # 반사(reflect) 패딩 적용
            C, H, W = img.shape                    # 패딩 후 크기 다시 업데이트

        y0 = random.randint(0, H - ps)             # crop 시작 y(0~H-ps)
        x0 = random.randint(0, W - ps)             # crop 시작 x(0~W-ps)

        return img[:, y0:y0 + ps, x0:x0 + ps]      # crop된 patch 반환([C,ps,ps])

    def _augment(self, patch: torch.Tensor) -> torch.Tensor:
        """
        ✅ 역할:
        - patch에 랜덤 flip/rot augmentation 적용

        ✅ 왜?
        - 데이터 다양성 증가 -> 일반화 향상
        - 특히 denoising은 기하 변환이 label을 바꾸지 않으므로 안전한 augmentation
        """
        if not self.augment:                       # augment=False면
            return patch                           # 그대로 반환

        if random.random() < 0.5:                  # 50% 확률로
            patch = torch.flip(patch, dims=[2])    # 좌우 flip (width 축)

        if random.random() < 0.5:                  # 50% 확률로
            patch = torch.flip(patch, dims=[1])    # 상하 flip (height 축)

        k = random.randint(0, 3)                   # 0~3 중 하나 -> 0/90/180/270도 회전
        patch = torch.rot90(patch, k=k, dims=[1, 2])  # H,W 축 기준 회전

        return patch                               # augmentation 된 patch 반환

    def __getitem__(self, idx):
        """
        ✅ 역할:
        - DataLoader가 요청할 때마다 1개 샘플을 생성해서 반환

        ✅ idx는 사실상 무시:
        - steps 기반에서는 "무한 샘플링"이 목적이라 idx대로 특정 이미지를 고정하지 않고
          매번 random.choice로 이미지를 골라 샘플 다양성을 극대화한다.
        """
        path = random.choice(self.image_paths)     # 이미지 경로를 랜덤 선택
        img = load_image(path, grayscale=self.grayscale)  # 이미지 로드 -> [C,H,W], [0,1]

        patch = self._random_crop(img)             # 랜덤 crop -> [C,ps,ps]
        patch = self._augment(patch)               # augmentation 적용

        coords = stratified_sample_coords(         # mask할 좌표 N개 생성
            self.patch_size,                       # patch_h
            self.patch_size,                       # patch_w
            self.n_masked                          # n_points
        )

        masked_patch, mask = apply_n2v_masking(    # blind-spot 치환 수행
            patch,                                 # 원본 patch(타겟도 이걸 씀)
            coords,                                # 치환할 좌표들
            radius=self.neighbor_radius            # 주변 반경
        )

        target = patch                              # N2V 타겟은 clean이 아니라 원본 noisy patch 자체

        return masked_patch, target, mask           # (입력, 타겟, 손실 마스크) 3개 반환


# ---------------------------
# 5) U-Net (depth=2)
# ---------------------------
class ConvBlock(nn.Module):
    """
    ✅ 역할:
    - U-Net을 구성하는 기본 블록: Conv -> BN -> ReLU

    ✅ 왜 BN?
    - 학습 안정화(분포 안정), 더 큰 lr 사용 가능, 수렴 도움
    - (주의) 아주 작은 batch에서는 BN이 불안정할 수 있음 -> 그땐 InstanceNorm/GroupNorm 고려
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()                         # nn.Module 초기화(필수)
        self.conv = nn.Conv2d(                     # 2D 컨볼루션 레이어 정의
            in_channels,                           # 입력 채널 수
            out_channels,                          # 출력 채널 수(특징맵 수)
            kernel_size=kernel_size,               # 커널 크기(보통 3)
            padding=padding,                       # 패딩(보통 1이면 same 유지)
            bias=False                             # BN이 bias 역할을 어느정도 대체하므로 bias=False로 흔히 둠
        )
        self.bn = nn.BatchNorm2d(out_channels)     # 채널별 정규화(BN)
        self.relu = nn.ReLU(inplace=True)          # 비선형 활성화(ReLU), inplace=True면 메모리 절약

    def forward(self, x: torch.Tensor):
        return self.relu(self.bn(self.conv(x)))    # conv -> bn -> relu 순으로 통과

class UNet(nn.Module):
    """
    ✅ 역할:
    - N2V denoising의 대표 backbone인 U-Net

    ✅ 왜 U-Net?
    - downsampling으로 큰 receptive field 확보(넓은 문맥)
    - skip connection으로 고주파/디테일 보존(denoise에서 중요)
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32, out_channels: int = 1, kernel_size: int = 3):
        super().__init__()                         # nn.Module 초기화
        padding = kernel_size // 2                 # kernel=3이면 padding=1 -> 해상도 유지

        # -------- encoder stage 0 --------
        self.enc0_1 = ConvBlock(in_channels, base_channels, kernel_size, padding)      # 입력 -> base 채널
        self.enc0_2 = ConvBlock(base_channels, base_channels, kernel_size, padding)    # base -> base
        self.pool1 = nn.MaxPool2d(2, 2)            # H,W를 1/2로 다운샘플 (stride=2)

        # -------- encoder stage 1 --------
        self.enc1_1 = ConvBlock(base_channels, base_channels * 2, kernel_size, padding)    # base -> 2*base
        self.enc1_2 = ConvBlock(base_channels * 2, base_channels * 2, kernel_size, padding)# 2*base -> 2*base
        self.pool2 = nn.MaxPool2d(2, 2)            # 다시 H,W를 1/2 -> 총 1/4

        # -------- bottleneck --------
        self.mid1 = ConvBlock(base_channels * 2, base_channels * 4, kernel_size, padding)  # 2*base -> 4*base
        self.mid2 = ConvBlock(base_channels * 4, base_channels * 4, kernel_size, padding)  # 4*base -> 4*base

        # -------- decoder stage 1 --------
        self.upconv1 = nn.ConvTranspose2d(         # 업샘플링(TransposeConv)으로 해상도 2배
            base_channels * 4,                     # 입력 채널(보틀넥 출력)
            base_channels * 2,                     # 출력 채널(encoder stage1과 맞추기)
            kernel_size=2, stride=2
        )
        self.dec1_1 = ConvBlock(base_channels * 4, base_channels * 2, kernel_size, padding) # concat으로 채널 4*base
        self.dec1_2 = ConvBlock(base_channels * 2, base_channels * 2, kernel_size, padding)

        # -------- decoder stage 0 --------
        self.upconv2 = nn.ConvTranspose2d(         # 업샘플링(해상도 2배)
            base_channels * 2,
            base_channels,
            kernel_size=2, stride=2
        )
        self.dec0_1 = ConvBlock(base_channels * 2, base_channels, kernel_size, padding)    # concat으로 채널 2*base
        self.dec0_2 = ConvBlock(base_channels, base_channels, kernel_size, padding)

        # -------- output --------
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)  # 1x1 conv로 원하는 채널로 사상

    def forward(self, x: torch.Tensor):
        # ---- encoder 0 ----
        enc0 = self.enc0_2(self.enc0_1(x))         # 두 번 conv 블록 통과(특징 추출)

        # ---- encoder 1 ----
        enc1 = self.pool1(enc0)                    # downsample(해상도 1/2)
        enc1 = self.enc1_2(self.enc1_1(enc1))      # 특징 추출

        # ---- bottleneck ----
        mid = self.pool2(enc1)                     # downsample(총 1/4)
        mid = self.mid2(self.mid1(mid))            # 깊은 특징 추출(넓은 RF)

        # ---- decoder 1 ----
        dec1 = self.upconv1(mid)                   # upsample(해상도 2배)
        dec1 = torch.cat([dec1, enc1], dim=1)      # skip concat: 채널 방향(dim=1)으로 합침
        dec1 = self.dec1_2(self.dec1_1(dec1))      # 복원 특징 정제

        # ---- decoder 0 ----
        dec0 = self.upconv2(dec1)                  # upsample(원래 해상도 복귀)
        dec0 = torch.cat([dec0, enc0], dim=1)      # skip concat
        dec0 = self.dec0_2(self.dec0_1(dec0))      # 최종 특징 정제

        return self.out_conv(dec0)                 # 1x1 conv로 출력(denoised 이미지)


# ---------------------------
# 6) Masked MSE loss
# ---------------------------
def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    ✅ 역할:
    - 오직 mask=1인 위치에서만 MSE를 계산하는 loss

    ✅ args:
    - pred   : [B,C,H,W] 모델 출력(denoise 결과)
    - target : [B,C,H,W] 원본 noisy patch
    - mask   : [B,1,H,W] loss 계산 위치만 1 (blind-spot positions)
    - eps    : 분모가 0 되는 것 방지

    ✅ 왜 이렇게?
    - N2V 핵심은 "모델이 정답 픽셀을 보지 않고" 주변 문맥으로 맞추게 하는 것
    - 따라서 loss를 전체 픽셀에서 계산하면(특히 입력에 정답 픽셀이 남아있으면) 꼼수 학습 위험 증가
    - blind-spot 위치만 계산해야 논문 아이디어가 성립
    """
    if mask.shape[1] == 1 and pred.shape[1] > 1:  # 채널이 3인데 mask가 1채널이면
        mask = mask.repeat(1, pred.shape[1], 1, 1) # mask를 [B,3,H,W]로 복제해 채널 맞춤

    mse_map = (pred - target) ** 2                # 픽셀별 제곱 오차 맵([B,C,H,W])
    masked = mse_map * mask                       # mask=1인 위치만 남기고 나머지는 0

    return masked.sum() / (mask.sum() + eps)      # 선택된 픽셀들만 평균 MSE로 정규화


# ---------------------------
# 7) Train (steps-based)
# ---------------------------
def train_n2v_steps(
    train_root: str,                              # 학습 이미지 폴더 경로(예: BSD400)
    grayscale: bool = True,                       # True면 1채널, False면 RGB 3채널
    patch_size: int = 64,                         # 학습 패치 크기(논문에서 흔히 64x64 사용)
    n_masked: int = 64,                           # patch당 mask 좌표 개수(논문에서 N으로 표기)
    neighbor_radius: int = 5,                     # blind-spot 치환 시 주변을 고를 범위
    base_channels: int = 32,                      # U-Net 기본 채널 수(작게: 가볍게, 크게: 성능↑/연산↑)
    batch_size: int = 128,                        # 배치 크기(메모리/속도/BN 안정성과 관련)
    lr: float = 4e-4,                             # AdamW 학습률
    weight_decay: float = 1e-5,                   # AdamW 정규화(가중치 감쇠)
    steps: int = 5000,                            # 총 optimizer update 횟수(학습 길이)
    log_every: int = 100,                         # 몇 step마다 평균 loss를 출력할지
    ckpt_path: str = "./n2v_self_ckpt.pth",       # 체크포인트 저장 경로
    seed: int = 0,                                # 랜덤 시드(재현성)
) -> nn.Module:
    """
    ✅ 역할:
    - steps 번 학습(optimizer update)하고 모델을 반환한다.
    - 학습 끝나면 체크포인트(모델/옵티마/하이퍼파라미터)를 저장한다.
    """
    random.seed(seed)                              # python random 시드 고정(랜덤 crop/마스크 좌표에 영향)
    np.random.seed(seed)                           # numpy random 시드(현재 코드에서는 주로 PSNR용)
    torch.manual_seed(seed)                        # torch random 시드(가중치 초기화 등에 영향)

    in_ch = 1 if grayscale else 3                  # 입력 채널 수 결정(흑백=1, RGB=3)

    dataset = N2VPatchDataset(                     # N2V 학습용 dataset 생성(패치+마스크를 즉석 생성)
        image_root_or_list=train_root,             # 학습 이미지 폴더(또는 리스트)
        patch_size=patch_size,                     # patch 크기
        n_masked=n_masked,                         # mask 픽셀 수
        grayscale=grayscale,                       # grayscale 여부
        neighbor_radius=neighbor_radius,           # 치환 반경
        augment=True,                              # 학습에는 augmentation을 켜는 편이 보통 이득
    )

    loader = DataLoader(                           # 미니배치로 뽑아주는 DataLoader
        dataset,                                   # 위에서 만든 dataset
        batch_size=batch_size,                     # 배치 크기
        shuffle=True,                              # 섞어서 다양한 조합을 만들기
        num_workers=0,                             # Mac MPS 환경에서 일단 0이 안전(멀티프로세싱 이슈 방지)
        drop_last=True                             # 마지막 배치가 batch_size보다 작으면 버림(BN 안정성↑)
    )

    model = UNet(                                  # U-Net 모델 생성
        in_channels=in_ch,                         # 입력 채널(1 or 3)
        base_channels=base_channels,               # 기본 채널 수
        out_channels=in_ch,                        # 출력 채널도 입력과 동일(denoised 이미지)
        kernel_size=3                              # conv kernel size
    ).to(device)                                   # 모델을 device로 이동(MPS/CUDA/CPU)

    optimizer = torch.optim.AdamW(                 # AdamW 옵티마이저(가중치 감쇠가 Adam과 분리)
        model.parameters(),                         # 학습할 파라미터들
        lr=lr,                                      # 학습률
        betas=(0.9, 0.999),                         # Adam 모멘텀 계수(일반 기본값)
        weight_decay=weight_decay                   # weight decay(정규화)
    )

    model.train()                                   # 학습 모드(BN/Dropout 등 동작이 train로 바뀜)

    it = iter(loader)                               # loader를 iterator로 만들어 next(it)로 배치를 계속 뽑음
    running = 0.0                                   # log_every 동안 누적 loss(평균 계산용)

    for step in range(1, steps + 1):                # 1부터 steps까지 학습 반복(=optimizer update 횟수)
        try:
            x_in, x_tgt, m = next(it)               # 배치 1개 가져오기(입력, 타겟, 마스크)
        except StopIteration:                       # loader를 한 바퀴 다 돌면 StopIteration 발생
            it = iter(loader)                       # iterator를 새로 만들어 다시 처음부터
            x_in, x_tgt, m = next(it)               # 첫 배치를 다시 가져옴

        x_in = x_in.to(device)                      # 입력 텐서를 device로 이동
        x_tgt = x_tgt.to(device)                    # 타겟 텐서를 device로 이동
        m = m.to(device)                            # 마스크 텐서를 device로 이동

        pred = model(x_in)                          # forward: denoised 예측값 생성
        loss = masked_mse_loss(pred, x_tgt, m)      # blind-spot 위치에서만 MSE 계산

        optimizer.zero_grad(set_to_none=True)       # 이전 step의 gradient 초기화(메모리 효율 ↑)
        loss.backward()                             # backward: gradient 계산
        optimizer.step()                            # update: 파라미터 갱신(= 1 step)

        running += loss.item()                      # loss 값을 python float로 가져와 누적

        if step % log_every == 0:                   # log_every마다 평균 loss 출력
            avg = running / log_every               # 평균 loss
            print(f"[step {step:5d}/{steps}] loss={avg:.6f}")  # 진행 상황 출력
            running = 0.0                           # 누적값 리셋

    torch.save({                                    # 체크포인트 파일로 저장(실험 재현을 위해 정보까지 저장)
        "model_state_dict": model.state_dict(),     # 모델 가중치(핵심)
        "optimizer_state_dict": optimizer.state_dict(),  # 옵티마 상태(재시작 학습에 필요)
        "steps": steps,                             # 몇 steps까지 학습했는지 기록
        "grayscale": grayscale,                     # 설정값 기록
        "in_ch": in_ch,                             # 입력 채널 기록
        "base_channels": base_channels,             # 네트워크 폭 기록
        "patch_size": patch_size,                   # 패치 크기 기록
        "n_masked": n_masked,                       # 마스크 포인트 수 기록
        "neighbor_radius": neighbor_radius,         # 치환 반경 기록
        "lr": lr,                                   # 학습률 기록
        "weight_decay": weight_decay,               # weight decay 기록
        "seed": seed,                               # 시드 기록
    }, ckpt_path)                                   # 저장 경로

    print(f"Saved checkpoint -> {ckpt_path}")       # 저장 완료 로그
    return model                                    # 학습된 모델 반환


# ---------------------------
# 8) Load checkpoint
# ---------------------------
def load_n2v_checkpoint(
    ckpt_path: str,                                 # 저장된 체크포인트 파일 경로
    device_override: Optional[torch.device] = None,  # 특정 디바이스로 강제 로드하고 싶을 때 사용
) -> nn.Module:
    """
    ✅ 역할:
    - 체크포인트를 읽어서 동일 구조의 모델을 만들고 state_dict를 로드해서 복원한다.

    ✅ args:
    - ckpt_path: 저장된 .pth 경로
    - device_override: torch.device("cpu") 같이 강제로 지정 가능
    """
    dev = device_override if device_override is not None else device  # override가 있으면 그걸 쓰고 아니면 global device
    ckpt = torch.load(ckpt_path, map_location=dev)                    # 체크포인트 로드(map_location으로 디바이스 매핑)

    in_ch = ckpt.get("in_ch", 1)                                      # 저장된 in_ch가 없으면 기본 1
    base_channels = ckpt.get("base_channels", 32)                     # 저장된 base_channels가 없으면 기본 32

    model = UNet(                                                     # 저장된 설정으로 동일 구조 모델 생성
        in_channels=in_ch,
        base_channels=base_channels,
        out_channels=in_ch,
        kernel_size=3
    ).to(dev)                                                         # 모델을 dev로 이동

    model.load_state_dict(ckpt["model_state_dict"])                   # 저장된 가중치를 모델에 로드
    model.eval()                                                      # eval 모드(추론 모드)로 전환(BN/Dropout 고정)

    print(f"Loaded checkpoint -> {ckpt_path} (device={dev})")         # 로드 로그
    return model                                                      # 복원된 모델 반환


# ---------------------------
# 9) Denoise + PSNR
# ---------------------------
@torch.no_grad()                                                      # 이 함수 안에서는 gradient 계산을 끔(추론 속도/메모리 ↑)
def denoise_image(
    model: nn.Module,                                                 # denoise에 사용할 모델(학습된 UNet)
    img_path: str,                                                    # 입력 이미지 파일 경로
    grayscale: bool = True,                                           # 입력이 grayscale인지 여부
    dev: Optional[torch.device] = None                                # 특정 디바이스로 추론하고 싶을 때
) -> np.ndarray:
    """
    ✅ 역할:
    - img_path 이미지를 모델에 넣어 denoise하고 uint8 이미지로 반환

    ✅ return:
    - np.ndarray uint8: grayscale면 [H,W], RGB면 [H,W,3]
    """
    dev = dev if dev is not None else next(model.parameters()).device # dev가 없으면 모델 파라미터가 올라간 디바이스 사용

    x = load_image(img_path, grayscale=grayscale)                     # [C,H,W] 로드
    x = x.unsqueeze(0).to(dev)                                        # 배치 차원 추가 -> [1,C,H,W] + device 이동

    model.eval()                                                      # 혹시 train 상태였을 수 있으니 eval로 고정
    pred = model(x).clamp(0, 1)                                       # 모델 출력 -> [1,C,H,W], 값 범위를 [0,1]로 clamp
    pred = pred[0].detach().cpu().numpy()                             # 배치 제거 -> [C,H,W], CPU numpy로 변환

    if pred.shape[0] == 1:                                            # 1채널이면
        pred = pred[0]                                                # [H,W]로 바꿈(저장 편의)
    else:
        pred = pred.transpose(1, 2, 0)                                # [C,H,W] -> [H,W,C]로 변환(이미지 저장 관례)

    return (pred * 255.0).astype(np.uint8)                            # [0,1] -> [0,255] uint8로 변환(이미지 저장 가능)

def psnr_uint8(
    pred_u8: np.ndarray,                                              # 예측 이미지(uint8)
    gt_u8: np.ndarray,                                                # GT 이미지(uint8)
    data_range: float = 255.0,                                        # 최대 값(8bit면 255)
    eps: float = 1e-12                                                # mse=0 방지용
) -> float:
    """
    ✅ 역할:
    - PSNR 계산 함수(단위 dB)

    PSNR = 10 * log10( (MAX^2) / MSE )
    - MSE가 작을수록 PSNR이 커짐(=좋은 복원)
    """
    pred = pred_u8.astype(np.float64)                                 # 계산 정확도 위해 float64 변환
    gt = gt_u8.astype(np.float64)                                     # GT도 float64로 변환

    mse = np.mean((pred - gt) ** 2)                                   # 평균 제곱 오차 계산
    if mse < eps:                                                     # 거의 0이면(완벽 일치)
        return float("inf")                                           # PSNR을 무한대로 처리

    return 10.0 * np.log10((data_range ** 2) / mse)                   # PSNR 공식 적용

def run_test_and_evaluate(
    model: nn.Module,                                                 # denoise에 사용할 모델
    test_root: str,                                                   # 테스트 데이터 root(그 아래 noisy/clean이 있다고 가정)
    out_dir: str = "./test_outputs",                                  # denoise 결과 저장 폴더
    grayscale: bool = True,                                           # grayscale 여부
    has_gt: bool = True,                                              # clean(GT) 폴더가 존재하는지 여부
    noisy_subdir: str = "noisy",                                      # noisy 폴더 이름
    clean_subdir: str = "clean",                                      # clean 폴더 이름
) -> None:
    """
    ✅ 역할:
    - test_root/noisy 안의 모든 이미지를 denoise하여 out_dir에 저장
    - has_gt=True이면 test_root/clean과 파일명 매칭해서 PSNR 계산

    ✅ args에서 "건네야 하는 것"
    - test_root: 아래 구조를 만족해야 함
      test_root/
        noisy/
          xxx.png ...
        clean/   (has_gt=True일 때)
          xxx.png ...
    """
    os.makedirs(out_dir, exist_ok=True)                               # 출력 폴더 생성(이미 있으면 에러 없이 통과)

    noisy_dir = os.path.join(test_root, noisy_subdir)                 # noisy 폴더 전체 경로 생성
    if not os.path.isdir(noisy_dir):                                  # noisy_dir이 실제 폴더인지 체크
        raise RuntimeError(f"noisy dir not found: {noisy_dir}")       # 없으면 즉시 에러(경로가 잘못됨)

    noisy_paths = list_images(noisy_dir)                              # noisy 폴더 내 이미지 파일 경로 전부 수집
    if len(noisy_paths) == 0:                                         # 이미지가 없으면
        raise RuntimeError(f"No images found in: {noisy_dir}")        # 테스트 불가

    clean_dir = os.path.join(test_root, clean_subdir) if has_gt else None  # GT를 쓸 거면 clean 폴더 경로 생성
    if has_gt and (clean_dir is None or not os.path.isdir(clean_dir)):     # has_gt=True인데 clean 폴더가 없으면
        raise RuntimeError(f"clean dir not found: {clean_dir}")            # 즉시 에러

    psnr_list: List[float] = []                                       # PSNR 값들을 모을 리스트

    for i, npath in enumerate(noisy_paths, 1):                        # noisy 이미지들을 1부터 번호 매겨 순회
        fname = os.path.basename(npath)                               # 파일명만 추출(예: "001.png")

        out_u8 = denoise_image(model, npath, grayscale=grayscale)     # noisy 이미지를 denoise -> uint8 결과
        save_path = os.path.join(out_dir, fname)                      # 저장할 경로(out_dir/파일명)
        Image.fromarray(out_u8).save(save_path)                       # PIL로 이미지 저장

        if has_gt:                                                    # GT가 있다면 PSNR 계산
            cpath = os.path.join(clean_dir, fname)                    # clean/파일명 경로 만들기(파일명 매칭 기준)
            if os.path.exists(cpath):                                 # GT 파일이 존재하면
                gt_img = Image.open(cpath)                            # GT 이미지 로드
                gt_img = gt_img.convert("L") if grayscale else gt_img.convert("RGB")  # 모드 맞추기
                gt_u8 = np.array(gt_img, dtype=np.uint8)              # numpy uint8로 변환
                psnr_list.append(psnr_uint8(out_u8, gt_u8))           # PSNR 계산해서 리스트에 저장
            else:
                print(f"[WARN] GT not found: {fname}")                # 매칭 실패하면 경고 출력

        if i % 50 == 0 or i == len(noisy_paths):                      # 50장마다 혹은 마지막이면 진행 로그 출력
            print(f"[{i}/{len(noisy_paths)}] saved -> {save_path}")   # 현재까지 저장된 파일 로그

    print("\n====================")                                     # 결과 요약 출력 시작
    print("TEST DONE")
    print(f"Saved outputs to: {out_dir}")

    if has_gt and len(psnr_list) > 0:                                 # GT가 있고 PSNR이 계산되었다면
        print(f"PSNR mean: {float(np.mean(psnr_list)):.4f} dB")       # 평균 PSNR
        print(f"PSNR std : {float(np.std(psnr_list)):.4f} dB")        # 표준편차
        print(f"PSNR min/max: {float(np.min(psnr_list)):.4f} / {float(np.max(psnr_list)):.4f} dB")  # min/max
    elif has_gt:                                                      # GT는 있다고 했는데 매칭이 안 됐다면
        print("PSNR: No matched GT pairs were found. (check filename matching rules)")
    else:                                                             # GT를 안 쓰는 테스트면
        print("PSNR: skipped (no GT provided)")
    print("====================\n")                                    # 결과 요약 끝


# ---------------------------
# 10) Main
# ---------------------------
if __name__ == "__main__":                                            # 이 파일을 직접 실행할 때만 아래 코드 실행(모듈 import시 실행 방지)
    # ✅ (A) Train
    # - train_root에 BSD400 폴더를 넣으면 그 안의 이미지로 패치들을 계속 뽑아서 학습한다.
    # model = train_n2v_steps(                                          # 학습 함수 호출 -> 모델 반환
    #     train_root="pytorch_n2v/denoising-datasets-main/BSD400",       # (필수) 학습 이미지 폴더 경로로 바꿔야 함
    #     grayscale=True,                                               # (선택) 흑백이면 True, RGB면 False
    #     patch_size=64,                                                # (선택) 패치 크기
    #     n_masked=64,                                                  # (선택) 패치당 mask 좌표 수
    #     neighbor_radius=5,                                            # (선택) 치환 반경
    #     base_channels=32,                                             # (선택) U-Net 폭(성능/속도 트레이드오프)
    #     batch_size=128,                                               # (선택) 배치 크기(메모리 한계에 맞춰 조절)
    #     lr=4e-4,                                                      # (선택) 학습률
    #     weight_decay=1e-5,                                            # (선택) weight decay
    #     steps=5000,                                                   # ✅ (핵심) 총 optimizer update 횟수
    #     log_every=100,                                                # (선택) 몇 step마다 로그 출력할지
    #     ckpt_path="./n2v_self_ckpt.pth",                              # (선택) 체크포인트 저장 위치
    #     seed=0,                                                       # (선택) 시드
    # )

    # ✅ (B) Load only (학습 생략하고 바로 체크포인트로부터 로드하고 싶으면)
    model = load_n2v_checkpoint("./n2v_self_ckpt.pth")               # 저장된 모델을 로드해서 바로 테스트/추론 가능

    # ✅ (C) Test + PSNR (BSD68 등 noisy/clean 쌍이 있을 때)
    run_test_and_evaluate(                                          # 테스트 실행(denoise 저장 + PSNR 평가)
        model=model,                                                # 사용할 모델
        test_root="pytorch_n2v/denoising-datasets-main/BSD68",       # 테스트 데이터 root 경로로 바꿔야 함
        out_dir="./test_outputs",                                   # 출력 저장 폴더
        grayscale=True,                                             # 데이터 채널과 맞추기
        has_gt=True,                                                # clean GT가 있으면 True
        noisy_subdir="noisy",                                       # noisy 폴더 이름(데이터 구조에 맞추기)
        clean_subdir="clean",                                       # clean 폴더 이름(데이터 구조에 맞추기)
    )
