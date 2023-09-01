import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import vizdoom as vzd
from collections import deque
import random
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, c1=8, c2=16, c3=24, encode_dims=128):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(c3*26*41, encode_dims)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encode_dims, c3*26*41),
            nn.ReLU(),
            nn.ConvTranspose2d(c3, c2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.ConvTranspose2d(c1, 1, kernel_size=3, stride=1, bias=False),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the autoencoder and optimizer
autoencoder = Autoencoder(encode_dims=64)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img.requires_grad_()

        # Forward pass
        output = autoencoder(img)

        # Compute the loss
        loss = criterion(output, img)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")