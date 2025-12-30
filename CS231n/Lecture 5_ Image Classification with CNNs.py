import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# Hyperparameters -> 우리가 학습 이전에 설정 가능한 값들
num_epochs = 10
batch_size = 64
learning_rate = 1e-4
num_classes = 10


transform = {
    'train' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),

    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

train_dataset = datasets.MNIST(root = './data', train = True, download = False, transform = transform['train'])
test_dataset = datasets.MNIST(root = './data', train = False, download = False, transform = transform['test'])

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)





class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()


        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 0) # stride를 키우면 Stride는 receptive field 확장과 계산 감소를 통해 전역적 문맥 학습을 가능하게 하고, padding은 출력 크기를 조절하고 경계 정보를 보존하는 역할을 함.
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0) # 주로 kernel_size = 2, stride = 2로 많이 사용하거나 혹릉 4, 2로 많이 사용, pooling은 학습되지 않은 요약을 통해 위치 불변성과 일반화된 표현을 제공한다.
        self.relu = nn.ReLU() # 비선형 활성화 함수 -> ReLU가 가장 많이 사용됨, 비선형성을 통해서 모델의 표현력을 높여줌
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 , 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x) # 보통 Conv -> Activation function -> Pooling 순으로 많이 사용
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out) # 마지막 fc레이어에서는 보통 활성화 함수를 사용하지 않음 (CrossEntropyLoss 내부에서 softmax를 포함하고 있기 때문)
        return out
    
model = CNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = learning_rate, betas = (0.9, 0.999), weight_decay = 5e-4) # Lecture4에서 배웠듯이 Adam에서 정규화항을 더해서 수행하고 싶으면 AdamW를 사용해야 함. (Adam에서는 각 기울기의 따라 파라미터 별로 서로 다른 기준으로 적용하지만 AdamW는 동일한 기준으로 모든 파라미터에 적용이 가능함.)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        ouputs = model(images)
        loss = criterion(ouputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

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
        

PATH = './cnn_cs231n.pth'
torch.save({
    'model_state_dict' : model.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
    'epoch': num_epochs
}, PATH)