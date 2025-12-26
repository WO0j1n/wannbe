print("--- RUNNING MODIFIED UNET_SELF.PY ---")
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
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive # Import this

class VOCSegmentationWithCustomURL(VOCSegmentation):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        self.root = root
        self.url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
        self.filename = 'VOCtrainval_11-May-2012.tar'
        self.md5 = '6cd6e2733973aa67476e330a6747466d'
        
        try:
            # Try to initialize, it will fail if data is not found
            super().__init__(root, year=year, image_set=image_set, download=False, transform=transform, target_transform=target_transform)
        except RuntimeError as e:
            # If it fails because the dataset is not found, and download is True, then we handle it.
            if 'Dataset not found' in str(e) and download:
                download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)
                # Now that it's downloaded, we need to re-initialize the parent class so it can find the files.
                super().__init__(root, year=year, image_set=image_set, download=False, transform=transform, target_transform=target_transform)
            else:
                # If it's a different error, or download is False, re-raise it.
                raise e


device = torch.device('mps' if torch.mps.is_available() else 'cpu')


NUM_CLASSES = 21                 # VOC: background 포함 21
IGNORE_INDEX = 255               # VOC void label
IMG_SIZE = 256                   # (H, W)로 resize
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 20
NUM_WORKERS = 0


transfoerm = {
    "image" : transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ]),

    "mask" : transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor(),  # uint8 tensor (1, H, W)
    ])
}


def preprocess_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    """
    mask_tensor: (1, H, W) uint8
    return     : (H, W) long
    """
    # (1,H,W) -> (H,W)
    mask = mask_tensor.squeeze(0).long()
    return mask

# =========================


train_dateset = VOCSegmentationWithCustomURL(
    root = "./data",
    year = "2012",
    image_set = "train",
    download = True,
    transform = transfoerm["image"],
    target_transform = transfoerm["mask"]
)

# Note: The 'test' set for VOC2012 is not publicly available with ground truth.
# People usually use the 'val' set for validation/testing.
# The download will fetch 'train' and 'val' together.
test_dataset = VOCSegmentationWithCustomURL(
    root = "./data",
    year = "2012",
    image_set = 'val',  # Changed from 'test' to 'val'
    download = True,
    transform = transfoerm["image"],
    target_transform = transfoerm["mask"]
)


def collate_fn(batch):
    imgs, masks = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    masks = torch.stack([preprocess_mask(m) for m in masks], dim=0)  # (B,H,W)
    return imgs, masks


train_loader = torch.utils.data.DataLoader(
    dataset = train_dateset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = NUM_WORKERS,
    pin_memory = False,
    collate_fn = collate_fn
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = NUM_WORKERS,
    pin_memory = False,
    collate_fn = collate_fn
)


# =========================


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

class UP(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear = True):
        super(UP, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
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
        self.up1 = UP(1024, 512 // factor, bilinear)  # concat: 512 + 512 = 1024
        self.up2 = UP(512, 256 // factor, bilinear)   # concat: 256 + 256 = 512
        self.up3 = UP(256, 128 // factor, bilinear)   # concat: 128 + 128 = 256
        self.up4 = UP(128, 64, bilinear)              # concat: 64 + 64 = 128

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
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            val_loss += loss.item()

            pred = torch.argmax(logits, dim=1)  # (B,H,W)
            val_miou += compute_miou(pred, masks, NUM_CLASSES, IGNORE_INDEX)

    val_loss /= len(test_loader)
    val_miou /= len(test_loader)

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