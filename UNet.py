import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

# =========================
# Device
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =========================
# Config
# =========================
ROOT = "./data"
YEAR = "2012"
NUM_CLASSES = 21                 # VOC: background 포함 21
IGNORE_INDEX = 255               # VOC void label
IMG_SIZE = 256                   # (H, W)로 resize
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 20
NUM_WORKERS = 0                  # mac/mps면 0이 편함

# =========================
# Transforms
# - image: float tensor + normalize
# - mask : long tensor (class index), NEAREST resize
# =========================
image_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

mask_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.NEAREST),
    T.PILToTensor(),  # uint8 tensor (1, H, W)
])

def preprocess_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    """
    mask_tensor: (1, H, W) uint8
    return     : (H, W) long
    """
    # (1,H,W) -> (H,W)
    mask = mask_tensor.squeeze(0).long()
    return mask

# =========================
# Dataset
# =========================
train_set = VOCSegmentation(
    root=ROOT,
    year=YEAR,
    image_set="train",
    download=True,
    transform=image_transform,
    target_transform=mask_transform
)

val_set = VOCSegmentation(
    root=ROOT,
    year=YEAR,
    image_set="val",
    download=False,
    transform=image_transform,
    target_transform=mask_transform
)

def collate_fn(batch):
    imgs, masks = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    masks = torch.stack([preprocess_mask(m) for m in masks], dim=0)  # (B,H,W)
    return imgs, masks

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn
)

# =========================
# UNet (fixed)
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            # upsample만 하고 conv에서 채널 조정
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # concat 후 conv: in_channels 그대로 들어감
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # transpose conv로 채널을 절반으로 줄인 뒤 concat
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: decoder feature, x2: encoder skip
        x1 = self.up(x1)

        # 크기 맞추기 (패딩)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Up의 in_channels는 "concat 후 채널 수"
        self.up1 = Up(1024, 512 // factor, bilinear)  # concat: 512 + 512 = 1024
        self.up2 = Up(512, 256 // factor, bilinear)   # concat: 256 + 256 = 512
        self.up3 = Up(256, 128 // factor, bilinear)   # concat: 128 + 128 = 256
        self.up4 = Up(128, 64, bilinear)              # concat: 64 + 64 = 128

        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024//factor

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        logits = self.out_conv(x)  # (B, C, H, W)
        return logits

# =========================
# Metrics: mIoU
# =========================
@torch.no_grad()
def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int):
    """
    pred  : (B,H,W) long
    target: (B,H,W) long
    """
    miou_list = []
    for cls in range(num_classes):
        # ignore pixels
        valid = target != ignore_index

        pred_c = (pred == cls) & valid
        targ_c = (target == cls) & valid

        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()

        if union == 0:
            continue
        miou_list.append(inter / union)

    if len(miou_list) == 0:
        return 0.0
    return float(np.mean(miou_list))

# =========================
# Train / Eval
# =========================
model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=True).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)                  # (B,C,H,W)
        loss = criterion(logits, masks)         # masks: (B,H,W)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- val ----
    model.eval()
    val_loss = 0.0
    val_miou = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            val_loss += loss.item()

            pred = torch.argmax(logits, dim=1)  # (B,H,W)
            val_miou += compute_miou(pred, masks, NUM_CLASSES, IGNORE_INDEX)

    val_loss /= len(val_loader)
    val_miou /= len(val_loader)

    print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
          f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  mIoU={val_miou:.4f}")

# =========================
# Save checkpoint
# =========================
PATH = "./unet_voc2012.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": EPOCHS,
}, PATH)

print(f"Saved: {PATH}")
