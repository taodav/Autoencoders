import torch
import torch.nn as nn
from torch.nn import functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        General Autoencoder. Takes in as input Encoder and Decoder layers.
        :param encoder: something that's nn.Module
        :param decoder: something that's also nn.Module.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.Linear(self.input_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input_tensor):
        input_tensor = input_tensor.view(-1, 28 * 28)
        encoded = F.relu(self.encoder(input_tensor))
        decoded = F.tanh(self.decoder(encoded))
        return encoded, decoded


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(VariationalAutoencoder, self).__init__()
        self.hidden_size = hidden_size

        self.encoder = ImageEncoder(in_channels, hidden_size, flatten=False)
        self.decoder = ImageDecoder(hidden_size, out_channels)

        self.mean_dense = nn.Linear(32, hidden_size)
        self.std_dense = nn.Linear(32, hidden_size)

    def forward(self, image):
        encoded = self.encoder(image)
        mean = self.mean_dense(encoded)
        log_var = self.std_dense(encoded)
        log_std = 0.5 * log_var
        if self.training:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            encoded = eps.mul(std).add_(mean)
        else:
            encoded = mean

        decoded = F.sigmoid(self.decoder(encoded))

        return encoded, decoded, mean, log_var
        

class ImageAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(ImageAutoencoder, self).__init__()

        self.hidden_size = hidden_size

        self.encoder = ImageEncoder(in_channels, hidden_size)
        self.decoder = ImageDecoder(hidden_size, out_channels)

    def forward(self, image):
        encoded, _ = self.encoder(image)
        decoded = F.sigmoid(self.decoder(encoded))
        return encoded, decoded, None, None


class ImageEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, flatten=True):
        super(ImageEncoder, self).__init__()

        self.flatten = flatten
        self.in_channels = in_channels
        self.hidden_size = hidden_size  # should be less than or equal to 8 * 3 * 3

        self.conv1 = nn.Conv2d(self.in_channels, 16, 3, stride=3, padding=1)  # b x 16 x 14 x 14
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # b x 32 x 3 x 3
        self.linear = nn.Linear(32, self.hidden_size)

    def forward(self, image):
        o = self.conv1(image)
        o = F.relu(F.max_pool2d(o, 2))
        o = self.conv2(o)
        o =F.relu(F.max_pool2d(o, 2))
        o = o.view(o.size(0), -1)  # b x 32
        if self.flatten:
            o = self.linear(o)

        return o


class ImageDecoder(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super(ImageDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.out_channels = out_channels

        self.convT1 = nn.ConvTranspose2d(32, 16, 5, stride=2)  # b x 16 x 5 x 5
        self.convT2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b x 8 x 15 x 15
        self.convT3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b x 1 x 28 x 28

    def forward(self, encoded):
        """
        decodes encoded state
        :param encoded: size b x hidden_size
        :return: b x 1 x 28 x 28 image
        """
        o = encoded.view(encoded.size(0), 32, 1, 1)
        o = F.relu(self.convT1(o))
        o = F.relu(self.convT2(o))
        o = self.convT3(o)

        return o




