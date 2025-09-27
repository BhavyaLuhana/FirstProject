import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transforms=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transforms=transforms.ToTensor()),
    batch_size=1000
)

class VAE(nn.module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.mu = nn.Linear(200, 10)
        self.logvar = nn.Linear(200, 10)
        self.fc2 = nn.Linear(10, 200)
        self.fc3 = nn.Linear(200, 28*28)

    def forward(self, x):
        h = F.relu(self.fc1(x.view(-1, 28*28)))
        mu, logvar = self.mu(h), self.logvar(h)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        h2 = F.relu(self.fc2(z))
        recon = torch.sigmoid(self.fc3(h2))
        return recon, mu, logvar
    
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    for data, _ in train_loader:
        recon, mu, logvar = model(data)
        bce = F.binary_cross_entropy(recon, data.view(-1, 28*28), reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = bce + kld

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Epoch" , epoch+1, "Loss: ", loss.item())

data, _ = next(iter(test_loader))
recon, _, _ = model(data)

fig, axes = ply.subplots(2, 8, figsize=(10, 3))
for i in range(8):
    axes[0, i].imshow(data[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon[i].view(28, 28).detach().numpy(), cmap='gray')
    axes[1, i].axis('off')
    plt.show()