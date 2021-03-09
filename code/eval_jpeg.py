"""Compress test data using jpeg and get mse and bpp.
To make things easy use the same config as the main train.py."""

from torchvision.io import encode_jpeg, decode_image
from torchvision.transforms import ConvertImageDtype, ToPILImage, ToTensor
import argparse
from train import get_dataloaders, get_ae_config
import helpers
import torch
import torch.nn.functional as F


def eval_jpeg(config):
    train_loader, test_loader = get_dataloaders(config)
    makeint = ConvertImageDtype(torch.uint8)
    topil = ToPILImage()
    totensor = ToTensor()
    quality = [1, 5, 10, 20, 40, 60, 80]
    mses = {}
    bpps = {}
    with torch.no_grad():
        for q in quality:
            n_samples = 0.
            mse = 0.
            bpp = 0.
            for batch in test_loader:
                batch = batch[0]
                n, c, h, w = batch.shape
                n_samples += n
                batch_int = makeint(batch)
                batch_jpeg = torch.zeros_like(batch)
                batch_bpp = torch.zeros(n)
                for idx in range(n):
                    jpeg = encode_jpeg(batch_int[idx], quality=q)
                    jpeg_bytes = len(jpeg)
                    batch_bpp[idx] = jpeg_bytes*8 / (h * w)
                    batch_jpeg[idx] = totensor(topil(decode_image(jpeg)))
                mse += F.mse_loss(batch, batch_jpeg).item()*n
                bpp += batch_bpp.sum().item()
            mses[q] = mse / n_samples
            bpps[q] = bpp / n_samples
            print(f"JPEG quality={q}, mse={mses[q]}, bpp={bpps[q]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfile", type=str, default="./configs/ae.ini",
                        help="Config file for the particular run.")
    args = parser.parse_args()
    config = get_ae_config(args.cfile)
    helpers.reproducibility_init(config.random_seed)
    eval_jpeg(config)
