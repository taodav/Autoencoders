import torch
import torch.nn as nn
from torch.nn import functional as F

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """
        General Autoencoder. Takes in as input Encoder and Decoder layers.
        :param encoder: something that's nn.Module
        :param decoder: something that's also nn.Module.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor):
        encoded = F.tanh(self.encoder(input_tensor))
        decoded = F.tanh(self.decoder(encoded))
        return encoded, decoded

class ImageEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(ImageEncoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size  # should be less than or equal to 8 * 3 * 3

        self.conv1 = nn.Conv2d(self.in_channels, 16, 3, stride=3, padding=1)  # b x 16 x 10 x 10
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b x 8 x 3 x 3
        self.linear = nn.Linear(8 * 3 * 3, self.hidden_size)

    def forward(self, image):
        o = self.conv1(image)
        o = F.relu(F.max_pool2d(o, 2))
        o = self.conv2(o)
        o =F.relu(F.max_pool2d(o, 2))
        o = o.view(o.size(0), -1)
        logits = self.linear(o)

        return logits


class ImageDecoder(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super(ImageDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.out_channels = out_channels

        self.linear1 = nn.Linear(self.hidden_size, 8 * 3 * 3)
        self.convT1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b x 16 x 5 x 5
        self.convT2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b x 8 x 15 x 15
        self.convT3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b x 1 x 28 x 28

    def forward(self, encoded):
        """
        decodes encoded state
        :param encoded: size b x hidden_size
        :return: b x 1 x 28 x 28 image
        """
        o = self.linear1(encoded)
        o = F.relu(o.view(o.size(0), 8, 3, 3))
        o = F.relu(self.convT1(o))
        o = F.relu(self.convT2(o))
        o = F.relu(self.convT3(o))

        return o


class ImageAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(ImageAutoencoder, self).__init__()

        self.encoder = ImageEncoder(in_channels, hidden_size)
        self.decoder = ImageDecoder(hidden_size, out_channels)

    def forward(self, image):
        encoded = F.tanh(self.encoder(image))
        decoded = F.tanh(self.decoder(encoded))
        return encoded, decoded

