import torch
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from train import AutoencoderTrainer
from model import ImageAutoencoder, Autoencoder, VariationalAutoencoder
from utils import device

t = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.MNIST('./data',
                                     transform=t)
img, target = dataset[0]  # just to get dimensions

# model = Autoencoder(img.size(1) * img.size(2), 300).to(device)
# model = ImageAutoencoder(img.size(0), 32, 1).to(device)
model = VariationalAutoencoder(img.size(0), 32, 1).to(device)

trainer = AutoencoderTrainer(dataset, model)
trainer.train()
