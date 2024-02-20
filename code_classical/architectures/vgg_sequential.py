'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils import spectral_norm
from architectures.SpectralLinear import SpectralLinear
from architectures.SpectralConv import SpectralConv


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, output_dim, k_lipschitz=None, p_drop=None):
        super(VGG, self).__init__()
        self.features = features
        if k_lipschitz is not None:
            l_1, l_2, l_3 = SpectralLinear(512, 512, k_lipschitz), SpectralLinear(512, 512, k_lipschitz), SpectralLinear(512, output_dim, k_lipschitz)
            self.classifier = nn.Sequential(
                nn.Dropout(p=p_drop),
                l_1,
                nn.ReLU(True),
                nn.Dropout(p=p_drop),
                l_2,
                nn.ReLU(True),
                l_3,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=p_drop),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p=p_drop),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, output_dim),
            )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, k_lipschitz=None):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if k_lipschitz is not None:
                conv2d = SpectralConv(in_channels, v, kernel_dim=3, padding=1, k_lipschitz=k_lipschitz)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(output_dim, k_lipschitz=None, p_drop=.5):
    """VGG 11-layer model (configuration "A")"""
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 11.)
    return VGG(make_layers(cfg['A'], k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)


def vgg11_bn(output_dim, k_lipschitz=None, p_drop=.5):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 11.)
    return VGG(make_layers(cfg['A'], batch_norm=True, k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)


def vgg13(output_dim, k_lipschitz=None, p_drop=.5):
    """VGG 13-layer model (configuration "B")"""
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 13.)
    return VGG(make_layers(cfg['B'], k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)


def vgg13_bn(output_dim, k_lipschitz=None, p_drop=.5):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 13.)
    return VGG(make_layers(cfg['B'], batch_norm=True, k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)


def vgg16(output_dim, k_lipschitz=None, p_drop=.5):
    """VGG 16-layer model (configuration "D")"""
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 16.)
    return VGG(make_layers(cfg['D'], k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)


def vgg16_bn(output_dim, k_lipschitz=None, p_drop=.5):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 16.)
    return VGG(make_layers(cfg['D'], batch_norm=True, k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)


def vgg19(output_dim, k_lipschitz=None, p_drop=.5):
    """VGG 19-layer model (configuration "E")"""
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 19.)
    return VGG(make_layers(cfg['E'], k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)


def vgg19_bn(output_dim, k_lipschitz=None, p_drop=.5):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 19.)
    return VGG(make_layers(cfg['E'], batch_norm=True, k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)
