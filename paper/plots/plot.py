# Plotting of results for paper

import matplotlib.pyplot as plt
import pandas as pd
import math

ae = pd.read_csv("softae.csv")
vq = pd.read_csv("softvqae.csv")

fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(5,2))

##############################
# VQ vs SQ with alpha = 0, length of uncompressed message approx 256 elements
dt = vq[vq.alpha == 0]
dt = dt.sort_values("c_num")
dtt = dt[dt.dsf == 2]
dtemp = dtt[dtt.z_channels == 8]
l0 = axs[0, 0].plot(dtemp.Bits_per_pixel, dtemp.test_mse, marker=".", linestyle="--", c="#1f77b4", label="vq, 8")
dtemp = dtt[dtt.z_channels == 16]
l1 = axs[0, 0].plot(dtemp.Bits_per_pixel, dtemp.test_mse, marker=".", linestyle="--", c="#d62728", label="vq, 16")

dt = ae[ae.alpha == 0]
dt = dt.sort_values("c_num")
dtt = dt[dt.dsf == 8]
dtemp = dtt[dtt.z_channels == 8]
l2 = axs[0, 0].plot(dtemp.Bits_per_pixel, dtemp.test_mse, marker=".", linestyle=":", c="#1f77b4", label="sq, 8")
dtemp = dtt[dtt.z_channels == 16]
l3 = axs[0, 0].plot(dtemp.Bits_per_pixel, dtemp.test_mse, marker=".", linestyle=":", c="#d62728", label="sq, 16")
axs[0, 0].legend()
axs[0, 0].set_xlabel("bpp")
axs[0, 0].set_ylabel("mse")
axs[0, 0].set_title("VQ vs SQ")
##############################


##############################
# VQ at different levels
dt = vq[vq.z_channels == 8]
dt = dt.sort_values("alpha")
dtt = dt[dt.dsf == 2]
dtemp = dtt[dtt.c_num == 8]
axs[0, 1].plot(dtemp.Bits_per_pixel, dtemp.test_mse, marker=".", linestyle=":", c="#1f77b4", label=f"k=8")
dtemp = dtt[dtt.c_num == 32]
axs[0, 1].plot(dtemp.Bits_per_pixel, dtemp.test_mse, marker=".", linestyle=":", c="#d62728", label=f"k=32")
dtemp = dtt[dtt.c_num == 128]
axs[0, 1].plot(dtemp.Bits_per_pixel, dtemp.test_mse, marker=".", linestyle=":", c="#2ca02c", label=f"k=128")
axs[0, 1].legend()
axs[0, 1].set_xlabel("bpp")
axs[0, 1].set_ylabel("mse")
axs[0, 1].set_title("Effect of alpha")
##############################

plt.show()
