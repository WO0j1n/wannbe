import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# Hyperparameters
num_epochs = 100
batch_size = 16
learning_rate = 0.001
num_classes = 1000

transform = { # 다양한 Augmentation 기법들을 transforms로 활용 가능함.
    'train' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
}

# ImageNet의 경우 직접 다운로드를 받아야 하며 파이토치에서 제공하지 않지만 해당 Lecture의 예제를 위해서 다음과 같이 수행했습니다.
train_dataset = datasets.ImageNet(root = './data', train = True, download = True, transform = transform['train'])
test_dataset = datasets.ImageNet(root = './data', train = False, download = True, transform = transform['test'])

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.direct = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Normalization 기법 중 하나로, 배치 단위로 입력 데이터를 정규화하여 학습을 안정화하고 속도를 향상시키는 역할을 함.
            # BatchNormd은 배치에 들어 있는 모든 이미지들에 대해, 같은 채널(C)의 모든 픽셀(H×W)을 모아서 평균·분산을 계산
            # LayerNormd은 이미지 한 장에 대해, 각 채널(C)의 모든 픽셀(H×W)을 모아서 평균·분산을 계산
            
            nn.GELU(),  # ReLU 대신 GELU 사용 (원하면 nn.ReLU(inplace=True)로 바꿔도 됨) -> 해당 Lecture에서 ReLU 대신 GELU를 사용함으로써 더 부드러운 활성화 함수를 통해 모델의 표현력을 향상시킬 수 있다고 함.
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity() # 입력으로 들어온 텐서를 아무런 변화 없이 패스시키는 레이어로 y = x가 됨

        self.act = nn.GELU()

    def forward(self, x):
        out = self.direct(x)
        out = out + self.shortcut(x)
        out = self.act(out)
        return out

# ---------------------------
# ResNet-18 (ImageNet style)
# ---------------------------
class ResNet18(nn.Module):
    def __init__(self, block, num_classes=1000):
        super().__init__()

        self.in_channels = 64

        # ✅ ResNet 1st conv block: 7x7 stride2 + maxpool
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # padding=3이 표준
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet18 stage: [2,2,2,2]
        self.layer1 = self.make_layer(block, 64,  2, stride=1) # out_channels = 64, num_blocks = 2, stride = 1
        self.layer2 = self.make_layer(block, 128, 2, stride=2) # out_channels = 128, num_blocks = 2, stride = 2 (처음에만 stride 2, 그 이후로는 stride - 1)
        self.layer3 = self.make_layer(block, 256, 2, stride=2) # out_channels = 256, num_blocks = 2, stride = 2 (처음에만 stride 2, 그 이후로는 stride - 1)
        self.layer4 = self.make_layer(block, 512, 2, stride=2) # out_channels = 512, num_blocks = 2, stride = 2 (처음에만 stride 2, 그 이후로는 stride - 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # AdaptiveAvgPool2d(1,1)은 입력 feature map의 크기와 상관없이 각 채널을 전역 평균 풀링하여 1×1로 축소함으로써 Global AveragePooling 수행
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



model = ResNet18(ResidualBlock).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = learning_rate, betas = (0.9, 0.999), weight_decay= 1e-2)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1 % 100) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/]{total_step}], Loss: {loss.item():.4f}')

# Testing Loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total} %')

PATH = './resnet_cs231n.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch' : num_epochs
}, PATH)