import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision
from torchvision import datasets, transforms


device = torch.device('mps' if torch.mps.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 1e-4
hidden_size = 256
hidden_size2 = 128
num_classes= 10

transform = {
    "train" : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),

    "test" : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

train_dataset = datasets.CIFAR10(root = "./data", train = True, download = False, transform = transform['train'])
test_dataset = datasets.CIFAR10(root = "./data", train = False, download = False, transform = transform['test'])

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class NeuralNet(nn.Module):
    def __init__(self, hidden_size, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
model = NeuralNet(hidden_size, hidden_size2, num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas = (0.9, 0.99), weight_decay=1e-2) #Adam의 경우 정규화항을 함께 활용하는 데 있어서 한계점이 존재(모든 파라미터들의 동일한 기준으로 줄이는 것이 불가능 함) 이를 해결하고자 나온 것이 AdamW로 해당 기법에서는 정규화항인 weight_decay를 적용시킬 수 있음.


# Training Loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Testing Loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total} %')

# Save the model checkpoint
PATH = './cs231n_nn.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch' : num_epochs
}, PATH)