import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from train import AutoencoderTrainer

dataset = torchvision.datasets.MNIST('./data')

trainer = AutoencoderTrainer(dataset)
trainer.train()

image, label = dataset[0]
plt.imshow(np.asarray(image))
plt.show()