import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# =========================
# Device (Apple Silicon MPS)
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =========================
# Hyperparameters
# =========================
img_size = 28 * 28
noise_dim = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024

batch_size = 64
lr = 1e-4
num_epochs = 10
D_steps = 2  # Discriminator steps per G step

# =========================
# Models
# =========================
class Discriminator(nn.Module):
    """
    MLP Discriminator for MNIST (flattened 28*28 input)
    Output: logits (NOT sigmoid), for BCEWithLogitsLoss
    """
    def __init__(self, img_size, hidden_size1, hidden_size2, hidden_size3):
        super().__init__()
        self.fc1 = nn.Linear(img_size, hidden_size3)
        self.fc2 = nn.Linear(hidden_size3, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size1)
        self.fc4 = nn.Linear(hidden_size1, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)  # logits
        return x


class Generator(nn.Module):
    """
    MLP Generator for MNIST
    Input: noise (batch, noise_dim)
    Output: fake image in [-1, 1] range (tanh), shape (batch, 28*28)
    """
    def __init__(self, img_size, hidden_size1, hidden_size2, hidden_size3, noise_dim):
        super().__init__()
        self.fc1 = nn.Linear(noise_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, img_size)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        z = self.fc4(z)
        z = self.tanh(z)
        return z


discriminator = Discriminator(img_size, hidden_size1, hidden_size2, hidden_size3).to(device)
generator = Generator(img_size, hidden_size1, hidden_size2, hidden_size3, noise_dim).to(device)

# =========================
# Loss + Optimizers
# =========================
# Use BCEWithLogitsLoss for numerical stability (no sigmoid in D)
criterion = nn.BCEWithLogitsLoss()

d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

# StepLR: step per epoch (NOT per iteration)
d_lr_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=30, gamma=0.1)
g_lr_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.1)

# =========================
# Data
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST: 1 channel
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# =========================
# Training
# =========================
for epoch in range(num_epochs):
    discriminator.train()
    generator.train()

    for i, (images, _) in enumerate(train_loader):  # MNIST returns (image, label)
        # -------------------------
        # Prepare real images
        # -------------------------
        bs = images.size(0)
        real_images = images.view(bs, -1).to(device)  # (bs, 784)

        # Label smoothing (optional but often stabilizes D a bit)
        real_labels = torch.ones(bs, 1, device=device) * 0.9
        fake_labels = torch.zeros(bs, 1, device=device)

        # ============================================================
        # 1) Train Discriminator: maximize log D(x) + log(1 - D(G(z)))
        # ============================================================
        for _ in range(D_steps):
            noise = torch.randn(bs, noise_dim, device=device)
            fake_images = generator(noise)  # (bs, 784)

            # D on real
            real_logits = discriminator(real_images)
            d_real_loss = criterion(real_logits, real_labels)

            # D on fake (detach to avoid updating G here)
            fake_logits = discriminator(fake_images.detach())
            d_fake_loss = criterion(fake_logits, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # ============================================================
        # 2) Train Generator: maximize log D(G(z))  (or minimize BCE with real labels)
        # ============================================================
        # Re-sample noise (optional but common)
        noise = torch.randn(bs, noise_dim, device=device)
        fake_images = generator(noise)

        g_logits = discriminator(fake_images)
        g_loss = criterion(g_logits, torch.ones(bs, 1, device=device))  # want D to say "real"

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Step [{i+1}/{len(train_loader)}] "
                f"d_loss: {d_loss.item():.4f} "
                f"g_loss: {g_loss.item():.4f}"
            )

    # Step LR schedulers per epoch
    d_lr_scheduler.step()
    g_lr_scheduler.step()

print("Training finished.")

PATH = './gan_self.pth'
torch.save(
    {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    },
    PATH
)