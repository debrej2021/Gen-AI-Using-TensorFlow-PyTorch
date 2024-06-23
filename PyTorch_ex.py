# Install necessary packages if not already installed
# !pip install torch torchvision matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
num_epochs = 50
latent_size = 100

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = dsets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator model
class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Initialize models
G = Generator(latent_size)
D = Discriminator()

# Loss and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=learning_rate)
optimizerG = optim.Adam(G.parameters(), lr=learning_rate)

# Function to plot generated images
def plot_generated_images(epoch, G, num_images=10):
    z = torch.randn(num_images, latent_size)
    fake_images = G(z).view(-1, 28, 28).detach().numpy()
    fig, axes = plt.subplots(1, num_images, figsize=(10, 1))
    for i in range(num_images):
        axes[i].imshow(fake_images[i], cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.show()

# Training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size = images.size(0)  # Get the current batch size (may be smaller at the end)
        
        # Train Discriminator
        D.zero_grad()
        real_images = images.view(-1, 28*28)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        optimizerD.step()

        # Train Generator
        G.zero_grad()
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizerG.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss_real + d_loss_fake:.4f}, g_loss: {g_loss:.4f}')
    
    # Plot generated images every 10 epochs
    if (epoch + 1) % 10 == 0:
        plot_generated_images(epoch + 1, G)

print('Training finished.')
