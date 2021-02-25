"""SoftVqAe models."""

from autoencoder import CompAe, Quantizer, Histogram
from collections import namedtuple
import torch.nn.functional as F
import torch
from helpers import soft_cross_entropy
import torch.nn as nn

AeOutput = namedtuple('AeOuput', ['zbar', 'zhard', 'symbols', 'softsymbols', 'z', 'xhat'])

QuantizerOutput = namedtuple('QuantizerOutput', ['zbar', 'zsoft', 'zhard', 'symbols', 'softsymbols'])


class SoftAe(CompAe):
    """Like autoencoder.CompAe but with soft symbols."""

    def __init__(self, config):
        super().__init__(config)
        self.quant = SoftQuantizer(config)
        self.emodel = SoftHistogram(config)

    def forward(self, data):
        eout = self.enc(data)
        qout = self.quant(eout)
        dout = self.dec(qout.zbar)
        return AeOutput(qout.zbar, qout.zhard, qout.symbols, qout.softsymbols, eout, dout)

    def loss_func(self, x, inputs):
        alpha, beta = self.config.alpha, self.config.beta
        zbar, zhard, symbols, softsymbols, z, xhat = inputs
        crossH = self.emodel.loss_func(symbols)
        mse = F.mse_loss(x, xhat)
        softcrossH = self.emodel.soft_loss_func(softsymbols)
        total = mse + alpha * softcrossH + beta * crossH
        return {'crossH': crossH, 'softcrossH': softcrossH, 'mse': mse, 'total': total}


class SoftQuantizer(Quantizer):
    """Like autoencoder.CompAe but with soft symbols."""

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
        return QuantizerOutput(zbar, softout, hardout, symbols.view(b, c, h, w), phisoft.view(b, c, h, w, -1))


class SoftHistogram(Histogram):
    """Like autoencoder.Histogram but adding soft_cross_entropy loss."""

    def soft_loss_func(self, targets):
        if not self.config.novq:
            targets = targets.permute(0, 2, 3, 1)
        targets = targets.flatten(start_dim=0, end_dim=-2)
        params = self.thetas.detach()
        params = params.expand_as(targets)
        return soft_cross_entropy(params, targets)


class SoftVqAe(SoftAe):
    """Like autoencoder.CompAe but with soft symbols."""

    def __init__(self, config):
        super().__init__(config)
        self.quant = VQuantizer(config)


class VQuantizer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.centers = nn.Embedding(config.c_num, config.z_channels)
        torch.nn.init.uniform_(self.centers.weight, a=config.c_min, b=config.c_max)
        # self.centers = nn.Parameter(torch.rand(config.c_num) * diff + config.c_min)

    def _to_bchw(data, b, c, h, w):
        data.view(data.view(b, h, w, c).permute(0, 3, 1, 2))

    def forward(self, data):
        b, c, h, w = data.shape
        data = data.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
        data = data[:, :, None].expand(-1, -1, self.config.c_num)
        centers = self.centers.weight.T[None, :, :].expand_as(data)
        dist = torch.norm(data - centers, 2, dim=1)
        phisoft = F.softmax(-self.config.sigma*dist, dim=-1)
        _, symbols = dist.min(dim=-1)
        softout = torch.sum(phisoft[:, None, :]*centers, dim=-1)
        hardout = self.centers(symbols)
        zbar = softout + (hardout - softout).detach()
        
        def _to_bchw(dt):
            dt = dt.view(b, h, w, -1).permute(0, 3, 1, 2)
            return dt
        qout = QuantizerOutput(*tuple(map(_to_bchw, (zbar, softout, hardout, symbols, phisoft))))
        return qout
