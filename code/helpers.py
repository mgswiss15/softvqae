import argparse
import re
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import torch
import numpy as np
import wandb


class FileParser(argparse.ArgumentParser):
    """Tweaking original ArgumentParser to read key value pairs from lines.
    Ignores lines containing text in double brackets [xxx] and starting with hash #.
    """

    def convert_arg_line_to_args(self, arg_line):
        if re.match('\[.*\]', arg_line) or re.match('#.*', arg_line):
            return ''
        return arg_line.split()


class CifarSmall(CIFAR10):
    """Cheat class to allow for working with smaller cifar which does not pass md5sum check."""

    def _check_integrity(self):
        return True


def grid_img(img):
    """Plot images in two rows. Expects images in 0-255 range."""
    # img = img / 255.
    grd = torchvision.utils.make_grid(img, nrow=img.shape[0]//2)
    fig = plt.imshow(grd.permute(1, 2, 0))
    return fig


def soft_cross_entropy(input, target):
    """Cross-entropy with soft targets, e.g. result of targets=softmax(h).
    Both input and target are (N, C).
    """

    logq = -F.log_softmax(input, dim=-1)
    ce = (target * logq).sum(dim=-1)
    return ce.mean()


def wandb_init(project, config):
    """Initialize wandb monitoring."""

    wandb.init(project=project)
    wandb.config.update(config)


def wandb_reinit(args):
    """Re-initialize wandb monitoring for additional evaluation."""

    id = args.wandb_run.rpartition('/')[2]
    wandb.init(project=f"{args.wandb_project}_{args.model}", resume="must", id=id)


def reproducibility_init(seed):
    """Initialize random state for reproducibility."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
