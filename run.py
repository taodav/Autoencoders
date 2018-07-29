import torch
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from train import AutoencoderTrainer
from model import ImageAutoencoder
from utils import device

t = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.MNIST('./data',
                                     transform=t)
img, target = dataset[0]  # just to get dimensions

model = ImageAutoencoder(img.size(0), 8 * 3 * 3, 1).to(device)

trainer = AutoencoderTrainer(dataset, model)
trainer.train()

image, label = dataset[0]
plt.imshow(np.asarray(image))
plt.show()