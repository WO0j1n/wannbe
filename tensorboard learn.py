import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# TensorBoard writer
writer = SummaryWriter('runs/mnist1')

device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print("Device:", device)

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 64
learning_rate = 0.001

# ----- Transforms -----
transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
    ])
}

# ----- Datasets & Dataloaders -----
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform['train'],
    download=False
)
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform['val'],
    download=False
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

# ----- 예시 이미지 TensorBoard에 기록 -----
example = iter(train_loader)
example_data, example_targets = next(example)

img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)

# ----- 모델 정의 -----
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = x.view(x.size(0), -1)  # [batch, 784]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----- Graph 기록 -----
# example_data: [N, 1, 28, 28]
writer.add_graph(model, example_data.to(device))

# ----- Training Loop -----
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # forward
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            avg_acc = running_correct / (100 * batch_size)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], '
                f'Step [{i + 1}/{n_total_steps}], '
                f'Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}'
            )

            # TensorBoard scalar 기록
            global_step = epoch * n_total_steps + i
            writer.add_scalar('training loss', avg_loss, global_step)
            writer.add_scalar('training accuracy', avg_acc, global_step)

            running_loss = 0.0
            running_correct = 0

    # ----- Evaluation / PR Curve -----
    model.eval()
    class_labels = []
    class_preds = []

    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            values, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            # 전체 배치에 대해 softmax → [batch, 10]
            class_probs_batch = F.softmax(outputs, dim=1)

            class_preds.append(class_probs_batch.cpu())
            class_labels.append(labels.cpu())

        class_preds = torch.cat(class_preds, dim=0)   # [N, 10]
        class_labels = torch.cat(class_labels, dim=0) # [N]

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc:.2f} %')

        # 클래스별 PR Curve 기록
        classes = range(10)
        for i in classes:
            labels_i = (class_labels == i)
            preds_i = class_preds[:, i]
            writer.add_pr_curve(str(i), labels_i, preds_i, global_step=epoch)

# 마지막에 한 번만
writer.close()
