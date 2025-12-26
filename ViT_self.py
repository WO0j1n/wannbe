import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm



device = torch.device('mps' if torch.mps.is_available() else 'cpu')


embed_size = 64 # 64, 128, 256 -> embed_size % num_heads == 0 이어야 함.
num_heads = 4
in_channels = 3
num_encoders = 6 # 논문에서는 12, 6, 24 층을 사용, 실험적으로 수행할 것
img_size = (28, 28)
patch_size = 7 # image_size % patch_size == 0 이어야 함.
num_classes = 10
epochs = 20

class Encoder(nn.Module):
    def __init__(self, embed_size, num_heads, dropout = 0.1):
        super(Encoder, self).__init__()
        self.attn = nn.MultiheadAttention(embed_size, num_heads, dropout = dropout, batch_first = True)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size), # ViT 논문에서 FFN의 hidden layer 크기는 embed_size의 4배
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # ✅ need_weights=False 추가 -> attention 연산 시 return 값이 attention out + atten_weights 두 개가 됨
        attn_out, _ = self.attn(x, x, x, need_weights=False)  # atten_weights는 학습에 필수적인 요소가 아니기에 False로 설정
        x = self.ln1(x)

        fc_out = self.fc(x)
        x = x + fc_out
        x = self.ln2(x)

        return x
    

class ViT(nn.Module):
    def __init__(self, in_channels, num_encoders, embed_size,
                img_size, patch_size, num_classes, num_heads):
        super(ViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        num_tokens = (img_size[0] * img_size[1]) // (patch_size ** 2)

        # ✅ 배치 차원 포함한 정석 shape
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_size)) # nn.Parameter()로 선언하여 학습 가능한 파라미터로 설정
        
        # ViT에서는 입력 크기와 패치 개수가 고정되어 있기 때문에, positional embedding을 주기 함수로 고정하지 않고 랜덤 초기화된 학습 가능한 파라미터로 두는 것이 일반적
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens + 1, embed_size))

         # in_channels = channel 수 * patch_size * patch_size(H*W), embeding_size 길이로 토큰화 하는 것이 Patch_Embedding
        self.patch_embedding = nn.Linear(in_channels * patch_size ** 2, embed_size)

        self.encoders = nn.ModuleList( # Encoder 여러 개 쌓을 땐ㄴ ModuleList를 통해 반복문을 활용하여 쌓음
            [Encoder(embed_size=embed_size, num_heads=num_heads) for _ in range(num_encoders)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 안전장치 (나중에 resize 넣으면 바로 잡아줌)
        assert H == self.img_size[0] and W == self.img_size[1], f"Expected {self.img_size}, got {(H,W)}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, -1, C * self.patch_size * self.patch_size)

        x = self.patch_embedding(patches)  # (Batch, Num_tokens, Embed_size)

        cls = self.class_token.repeat(B, 1, 1)  # (B, 1, E)
        x = torch.cat([cls, x], dim=1)          # (B, N+1, E)

        x = x + self.pos_embedding              # ✅ (1, N+1, E)라서 항상 안전, 입력으로 들어오는 patch의 위치 정보를 저장할 수 있음

        for encoder in self.encoders:
            x = encoder(x)

        cls_token = x[:, 0, :]
        out = self.mlp_head(cls_token)
        return out


model = ViT(
    in_channels = in_channels,
    num_encoders = num_encoders,
    embed_size = embed_size,
    img_size = img_size,
    patch_size = patch_size,
    num_classes = num_classes,
    num_heads = num_heads
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)


# 4) 데이터셋/로더 동일
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ✅ MNIST를 RGB처럼 3채널로
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dl  = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)


for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_dl):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dl.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

PATH = './vit_mnist.pth'

torch.save(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch' : epoch,
    },
    PATH
)