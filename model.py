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

        self.ngf = 32
        self.ndf = 32

        self.encoder = ImageEncoder(in_channels, self.ndf, hidden_size, flatten=False)
        self.decoder = ImageDecoder(hidden_size, self.ngf, out_channels)

        self.mean_dense = nn.Linear(self.ngf * 4 * 3 * 3, hidden_size)
        self.std_dense = nn.Linear(self.ngf * 4 * 3 * 3, hidden_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(mu) * std
            return mu + eps
        else:
            return mu

    def forward(self, image):
        encoding_1 = self.encoder(image)
        mean = self.mean_dense(encoding_1)
        log_var = self.std_dense(encoding_1)

        encoded = self.reparameterize(mean, log_var)

        decoded = F.sigmoid(self.decoder(encoded))

        return encoded, decoded, mean, log_var
        

class ImageAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(ImageAutoencoder, self).__init__()

        self.hidden_size = hidden_size

        self.encoder = ImageEncoder(in_channels, 32, hidden_size)
        self.decoder = ImageDecoder(hidden_size, 32, out_channels)

    def forward(self, image):
        encoded, _ = self.encoder(image)
        decoded = F.sigmoid(self.decoder(encoded))
        return encoded, decoded, None, None


class ImageEncoder(nn.Module):
    def __init__(self, in_channels, ndf, hidden_size, flatten=True):
        super(ImageEncoder, self).__init__()

        self.ndf = ndf

        self.flatten = flatten
        self.in_channels = in_channels
        self.hidden_size = hidden_size  # should be less than or equal to 8 * 3 * 3

        self.conv1 = nn.Conv2d(self.in_channels, self.ndf, 4, stride=2, padding=1)  # b x 32 x 14 x 14
        self.bn1 = nn.BatchNorm2d(self.ndf)

        self.conv2 = nn.Conv2d(self.ndf, self.ndf * 2, 4, stride=2, padding=1)  # b x 64 x 3 x 3
        self.bn2 = nn.BatchNorm2d(self.ndf * 2)

        self.conv3 = nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, stride=2, padding=1)  # b x 128 x 3 x 3
        self.bn3 = nn.BatchNorm2d(self.ndf * 4)

        if self.flatten:
            self.linear = nn.Linear(self.ndf * 4 * 3 * 3, self.hidden_size)

    def forward(self, image):
        o = F.relu(self.bn1(self.conv1(image)))  # b x ndf x 14 x 14
        o = F.relu(self.bn2(self.conv2(o)))  # b x ndf * 2 x 7 x 7
        o = F.relu(self.bn3(self.conv3(o)))  # b x ndf * 4 x 3 x 3

        o = o.view(o.size(0), -1)  # b x hidden_size
        if self.flatten:
            o = self.linear(o)

        return o


class ImageDecoder(nn.Module):
    def __init__(self, hidden_size, ngf, out_channels):
        super(ImageDecoder, self).__init__()

        self.ngf = ngf
        self.hidden_size = hidden_size
        self.out_channels = out_channels

        self.linear = nn.Linear(self.hidden_size, self.ngf * 4 * 3 * 3)

        self.convT1 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, stride=2, padding=1)  # b x 16 x 5 x 5
        self.bn1 = nn.BatchNorm2d(self.ngf * 2)

        self.convT2 = nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, stride=2)  # b x 8 x 15 x 15
        self.bn2 = nn.BatchNorm2d(self.ngf)

        self.convT3 = nn.ConvTranspose2d(self.ngf, self.out_channels, 4, stride=2, padding=1)  # b x 1 x 28 x 28
        self.bn3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, encoded):
        """
        decodes encoded state
        :param encoded: size b x hidden_size
        :return: b x 1 x 28 x 28 image
        """
        o = self.linear(encoded)
        o = o.view(o.size(0), self.ngf * 4, 3, 3)  # b x ngf * 4 x 3 x 3
        o = F.relu(self.bn1(self.convT1(o)))  # b x ngf * 2 x 6 x 6
        o = F.relu(self.bn2(self.convT2(o)))  # b x ngf x 12 x 12
        o = F.relu(self.bn3(self.convT3(o)))

        return o




