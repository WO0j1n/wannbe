import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
from torchvision import datasets, transforms

class Encoder(nn.Module):
    def __init__(self, embed_size=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)

        ff_out = self.ff(x)
        x = x + ff_out
        x = self.ln2(x)

        return x


class ViT(nn.Module):
    def __init__(self, in_channels=3, num_encoders=6, embed_size=64,
                img_size=(28, 28), patch_size=7, num_classes=10, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        num_tokens = (img_size[0] * img_size[1]) // (patch_size ** 2)

        self.class_token = nn.Parameter(torch.randn(embed_size), requires_grad=True)
        self.patch_embedding = nn.Linear(in_channels * patch_size ** 2, embed_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(num_tokens + 1, embed_size),
            requires_grad=True
        )

        self.encoders = nn.ModuleList(
            [Encoder(embed_size=embed_size, num_heads=num_heads) for _ in range(num_encoders)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes),
        )

    def forward(self, x):
        B, C = x.shape[:2]

        patches = x.unfold(2, self.patch_size, self.patch_size) \
                  .unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(
            B, -1, C * self.patch_size * self.patch_size
        )  # (B, N_patches, C*P*P)

        x = self.patch_embedding(patches)  # (B, N_patches, E)

        class_token = self.class_token.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1)
        x = torch.cat([class_token, x], dim=1)  # (B, N_patches+1, E)

        x = x + self.pos_embedding.unsqueeze(0)  # (1, T, E) broadcast

        for encoder in self.encoders:
            x = encoder(x)

        cls_token = x[:, 0, :]  # (B, E)  ← squeeze()는 굳이 안 씀
        out = self.mlp_head(cls_token)  # (B, num_classes)

        return out


from torchinfo import summary
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = ViT(in_channels=1, img_size=(28, 28), patch_size=7, embed_size=64, num_heads=4, num_encoders=3).to(device)
summary(model, [2,1,28,28])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

from tqdm import tqdm

# 파이썬 3.10 이하에서 지원
# from torch_lr_finder import LRFinder
# lr_finder = LRFinder(model, optimizer, criterion, device="mps")
# lr_finder.range_test(train_dl, end_lr=100, num_iter=100)
# lr_finder.plot() # to inspect the loss-learning rate graph
# lr_finder.reset() # to reset the model and optimizer to their initial state
from torchinfo import summary

# 1) device 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using MPS")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

# 2) 모델 생성 + 디바이스 이동
model = ViT(
    in_channels=1,
    img_size=(28, 28),
    patch_size=7,
    embed_size=64,
    num_heads=4,
    num_encoders=3
).to(device)

# 3) summary도 같은 device에서 실행
summary(model, input_size=(2, 1, 28, 28), device=device)

# 4) 데이터셋/로더 동일
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dl  = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    losses = []
    print(f"Epoch {epoch+1} / {epochs}", end=" ")
    model.train()
    for i, (image, label) in enumerate(tqdm(train_dl)):
        image, label = image.to(device), label.to(device)

        # 디버그용: 처음 한 번만 찍어보기
        if epoch == 0 and i == 0:
            print("image device:", image.device)
            print("patch_emb device:", next(model.patch_embedding.parameters()).device)

        pred = model(image)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"loss: {sum(losses) / len(losses):.4f}", end=" ")

    model.eval()
    with torch.no_grad():
        cnt, correct_cnt = 0, 0
        for image, label in test_dl:
            image, label = image.to(device), label.to(device)
            pred = model(image).argmax(dim=1)
            cnt += label.shape[0]
            correct_cnt += (pred == label).sum().item()
        print("accuracy:", correct_cnt / cnt)

torch.save(model.state_dict(), './model.pt')
