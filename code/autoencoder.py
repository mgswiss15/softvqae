"""Replicates architecture of NN modules from Mentzer (2019)."""

import torch
import torch.nn as nn
from collections import namedtuple
import torch.nn.functional as F

AeOutput = namedtuple('AeOuput', ['zbar', 'zhard', 'symbols', 'z', 'xhat'])

QuantizerOutput = namedtuple('QuantizerOutput', ['zbar', 'zsoft', 'zhard', 'symbols'])


# Normalization not necessary - ImageNet32 is already normalized to [0, 1]

# def _get_mean_std(device):
#     """Use pre-calculated mean and std values to normalize the data."""
#     mean = torch.tensor([121.85369873, 113.58860779, 100.63715363]).to(device)[None, :, None, None]
#     std = torch.tensor([68.8940, 66.7393, 69.3703]).to(device)[None, :, None, None]
#     return mean, std


# def normalize(x, device):
#     mean, std = _get_mean_std(device)
#     return (x-mean)/std


# def denormalize(x, device):
#     mean, std = _get_mean_std(device)
#     return x*std - mean


class CompAe(nn.Module):
    """Compressive autoencoder as in Mentzer (2019).
    Simplified and stripped to the bare minimum.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._centers = None  # set in encode()
        self.enc = Encoder(config)
        self.dec = Decoder(config)
        self.quant = Quantizer(config)
        self.emodel = Histogram(config)

    def forward(self, data):
        eout = self.enc(data)
        qout = self.quant(eout)
        dout = self.dec(qout.zbar)
        return AeOutput(qout.zbar, qout.zhard, qout.symbols, eout, dout)

    def loss_func(self, x, inputs):
        beta = self.config.beta
        zbar, zhard, symbols, z, xhat = inputs
        crossH = self.emodel.loss_func(symbols)
        mse = F.mse_loss(x, xhat)
        # mse = ((x-xhat)**2).flatten(start_dim=1).sum(dim=1).mean()
        total = mse + beta * crossH
        return {'crossH': crossH, 'mse': mse, 'total': total}


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        n = config.ae_hidden
        k = config.ae_kernel
        p = (k+1)//2 - 1
        if config.dsf in [8, 4, 2]:
            self.l00 = nn.Sequential(
                nn.Conv2d(self.config.in_channels, n//2, kernel_size=k, stride=2, padding=p, bias=False),
                nn.BatchNorm2d(n//2),
                nn.ReLU())
        else:
            self.l00 = nn.Sequential(
                nn.Conv2d(self.config.in_channels, n//2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n//2),
                nn.ReLU())            
        if config.dsf == 8:
            self.l01 = nn.Sequential(        
                nn.Conv2d(n//2, n, kernel_size=k, stride=2, padding=p, bias=False),
                nn.BatchNorm2d(n),
                nn.ReLU()
                )
        else:
            self.l01 = nn.Sequential(        
                nn.Conv2d(n//2, n, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n),
                nn.ReLU()
                )
        self.resblocks = nn.ModuleList()
        for b in range(self.config.ae_resblocks):
            layers = []
            layers.append(ResBlock(num_conv2d=2, relu=True, in_channels=n, out_channels=n, kernel_size=3, padding=1))
            layers.append(ResBlock(num_conv2d=2, relu=True, in_channels=n, out_channels=n, kernel_size=3, padding=1))
            layers.append(ResBlock(num_conv2d=2, relu=True, in_channels=n, out_channels=n, kernel_size=3, padding=1))
            self.resblocks.append(nn.Sequential(*layers))
        self.resblocks.append(ResBlock(num_conv2d=2, relu=False, in_channels=n, out_channels=n, kernel_size=3, padding=1))
        if config.dsf in [8, 4]:
            self.l1 = nn.Sequential(
                nn.Conv2d(n, config.z_channels, kernel_size=k, stride=2, padding=p, bias=False),
                nn.BatchNorm2d(config.z_channels)
                )
        else:
            self.l1 = nn.Sequential(
                nn.Conv2d(n, config.z_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(config.z_channels)
                )

    def forward(self, data):
        # data = normalize(data, self.config.device)
        data = self.l00(data)
        data = self.l01(data)
        dataskip = data
        for b in range(self.config.ae_resblocks):
            data = self.resblocks[b](data) + data
        data = self.resblocks[b+1](data) + dataskip
        data = self.l1(data)
        return data


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        n = config.ae_hidden
        k = config.ae_kernel
        p = (k+1)//2 - 1
        if config.dsf in [8, 4]:
            self.l0 = nn.Sequential(
                nn.ConvTranspose2d(config.z_channels, n, kernel_size=k, stride=2, padding=p, bias=False),
                nn.BatchNorm2d(n),
                nn.ReLU(),
                )
        else:
            self.l0 = nn.Sequential(
                nn.ConvTranspose2d(config.z_channels, n, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n),
                nn.ReLU(),
                )
        self.resblocks = nn.ModuleList()
        for b in range(self.config.ae_resblocks):
            layers = []
            layers.append(ResBlock(num_conv2d=2, relu=True, in_channels=n, out_channels=n, kernel_size=3, padding=1))
            layers.append(ResBlock(num_conv2d=2, relu=True, in_channels=n, out_channels=n, kernel_size=3, padding=1))
            layers.append(ResBlock(num_conv2d=2, relu=True, in_channels=n, out_channels=n, kernel_size=3, padding=1))
            self.resblocks.append(nn.Sequential(*layers))
        self.resblocks.append(ResBlock(num_conv2d=2, relu=False, in_channels=n, out_channels=n, kernel_size=3, padding=1))
        if config.dsf == 8:
            self.l10 = nn.Sequential(
                nn.ConvTranspose2d(n, n//2, kernel_size=k, stride=2, padding=p, bias=False),
                nn.BatchNorm2d(n//2),
                nn.ReLU()
                )
        else:
            self.l10 = nn.Sequential(
                nn.ConvTranspose2d(n, n//2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n//2),
                nn.ReLU()
                )
        if config.dsf in [8, 4, 2]:
            self.l11 = nn.Sequential(
                nn.ConvTranspose2d(n//2, config.in_channels, kernel_size=k, stride=2, padding=p, bias=False),
                nn.BatchNorm2d(config.in_channels)
                )
        else:
            self.l11 = nn.Sequential(
                nn.ConvTranspose2d(n//2, config.in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(config.in_channels)
                )

    def forward(self, data):
        data = self.l0(data)
        dataskip = data
        for b in range(self.config.ae_resblocks):
            data = self.resblocks[b](data) + data
        data = self.resblocks[b+1](data) + dataskip
        data = self.l10(data)
        data = self.l11(data)
        # data = denormalize(data, self.config.device).clamp(0., 255.)
        return data.clamp(0., 1.)


class ResBlock(nn.Module):

    def __init__(self, num_conv2d, relu, **kwargs):
        super().__init__()
        layers = [nn.Conv2d(**kwargs)]
        for conv in range(num_conv2d-1):
            layers.append(nn.BatchNorm2d(kwargs['out_channels']))
            if relu:
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(**kwargs))
        layers.append(nn.BatchNorm2d(kwargs['out_channels']))
        self.sequential = nn.Sequential(*layers)

    def forward(self, data):
        return self.sequential(data) + data


class Quantizer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.centers = nn.Embedding(config.c_num, 1)
        torch.nn.init.uniform_(self.centers.weight, a=config.c_min, b=config.c_max)
        # self.centers = nn.Parameter(torch.rand(config.c_num) * diff + config.c_min)

    def forward(self, data):
        b, c, h, w = data.shape
        data = data.view(b, -1)
        data = data[:, :, None].expand(-1, -1, self.config.c_num)
        centers = self.centers.weight.T[None, :, :].expand_as(data)
        dist = (data - centers)**2
        phisoft = F.softmax(-self.config.sigma*dist, dim=-1)
        _, symbols = dist.min(dim=-1)
        softout = torch.sum(phisoft*centers, dim=-1).view(b, c, h, w)
        hardout = self.centers(symbols).view(b, c, h, w)
        zbar = softout + (hardout - softout).detach()
        return QuantizerOutput(zbar, softout, hardout, symbols.view(b, c, h, w))


class Histogram(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        c_num = self.config.c_num
        self.thetas = nn.Parameter(torch.ones(c_num)/c_num)

    def loss_func(self, targets):
        targets = targets.flatten()
        params = self.thetas.expand(targets.shape[0], -1)
        return F.cross_entropy(params, targets)
