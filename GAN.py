import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# device 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

image_size = 28 * 28
noise_dim = 100

class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_size1=256, hidden_size2=512, hidden_size3=1024):
        super().__init__()
        self.linear1 = nn.Linear(image_size, hidden_size3)
        self.linear2 = nn.Linear(hidden_size3, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size1)
        self.linear4 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self, noise_dim=100, image_size=28*28,
                 hidden_size1=256, hidden_size2=512, hidden_size3=1024):
        super().__init__()
        self.linear1 = nn.Linear(noise_dim, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, image_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.tanh(x)
        return x


discriminator = Discriminator(image_size=image_size).to(device)
generator = Generator(noise_dim=noise_dim, image_size=image_size).to(device)

learning_rate = 1e-4
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST는 1채널
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

num_epoch = 10

for epoch in range(num_epoch):
    for i, (images, _) in enumerate(train_loader):
        batch_size = images.size(0)

        real_images = images.view(batch_size, -1).to(device)

        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # -----------------
        # 1. Generator 학습
        # -----------------
        g_optimizer.zero_grad()
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_images = generator(noise)
        g_loss = criterion(discriminator(fake_images), real_label)
        g_loss.backward()
        g_optimizer.step()

        # -----------------
        # 2. Discriminator 학습
        # -----------------
        d_optimizer.zero_grad()

        # 실제 이미지
        real_out = discriminator(real_images)
        d_real_loss = criterion(real_out, real_label)

        # 가짜 이미지 (detach로 G에는 gradient 안 가도록)
        fake_out = discriminator(fake_images.detach())
        d_fake_loss = criterion(fake_out, fake_label)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epoch}] Step [{i+1}/{len(train_loader)}] "
                  f"d_loss: {d_loss.item():.4f}  g_loss: {g_loss.item():.4f}")
