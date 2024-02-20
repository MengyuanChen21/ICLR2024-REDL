import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, remove_spectral_norm
# from architectures.SpectralNorm2D import SpectralNorm2D as spectral_norm

class SpectralConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_dim, padding, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_dim, padding=padding))

    def forward(self, x):
        y = self.k_lipschitz * self.spectral_conv(x)
        return y
