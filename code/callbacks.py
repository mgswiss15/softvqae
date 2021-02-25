"""Callbacks inspired by fastai."""

import wandb
import torch
import torch.nn.functional as F
# import utils.plots as plt
import os
from helpers import grid_img
import math


class Callback():
    """Base class for callbacks."""

    def init_learner(self, learner):
        self.learner = learner

    @property
    def class_name(self):
        return self.__class__.__name__.lower()


class Wandb(Callback):
    """Callbacks for interacting with wandb."""

    # def fit_begin(self):
    #     """Watch all model stats."""
    #     wandb.watch(self.learner.model, log="all", log_freq=100)

    def trainbatch_end(self):
        """Log train losses."""
        losses = self.learner.losses_train
        logdict = {'train_'+key: value[-1] for key, value in losses.items()}
        wandb.log(logdict)

    def epoch_end(self):
        """Log test losses."""
        losses = self.learner.losses_test
        logdict = {'test_'+key: value[-1] for key, value in losses.items()}
        wandb.log(logdict)


class Printouts(Callback):
    """Callbacks for printing progress notifications."""

    def fit_begin(self):
        print(f"Will train model for {self.learner.epochs} epochs.")
        print(f"Train set has {len(self.learner.train_loader)} ",
              f"batches of size {self.learner.train_loader.batch_size}.")
        print(f"Test set has {len(self.learner.test_loader)} ",
              f"batches of size {self.learner.test_loader.batch_size}.", flush=True)

    def epoch_begin(self):
        print(f"Training epoch {self.learner.epoch} ...", flush=True)

    def epoch_end(self):
        print(f"Losses: train = {self.learner.losses_train['total'][-1]}, ",
              f"test = {self.learner.losses_test['total'][-1]}.", flush=True)


class Scheduler(Callback):
    """Callbacks for schedulers."""

    def fit_begin(self):
        optimizer = self.learner.optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR
        max_lr = self.learner.model.config.max_lr
        epochs = self.learner.epochs
        spe = len(self.learner.train_loader)
        self.scheduler = scheduler(optimizer, max_lr, epochs=epochs, steps_per_epoch=spe, final_div_factor=1000)

    def after_optimizer(self):
        self.scheduler.step()

    def trainbatch_end(self):
        """Log schedule."""
        optimizer = self.learner.optimizer
        lr = optimizer.param_groups[0]['lr']
        logdict = {'lr': lr}
        wandb.log(logdict)


class Evaluate(Callback):
    """Callback for final evaluation."""

    def fit_end(self):
        """Init data."""
        device = next(self.learner.model.parameters()).device
        data = next(iter(self.learner.test_loader))[0].to(device)
        data = data[:5, ...]
        self.learner.model.eval()
        with torch.no_grad():
            out = self.learner.model(data)

        # plot reconstuctions
        data = torch.cat((data, out.xhat), dim=0).to("cpu")
        fig = grid_img(data)
        wandb.log({f"Reconstructions": wandb.Image(fig)})
        # bits per pixel, centers distribution
        xdim = data[0].numel()
        zdim = out.z[0].numel()
        if self.learner.model.config.novq:
            bpp = self.learner.losses_test['crossH'][-1]*zdim / (math.log(2.)*xdim) * 3  # treat all channels as a single pixel
        else:
            bpp = self.learner.losses_test['crossH'][-1]*(zdim / out.z[0].shape[1]) / (math.log(2.)*xdim) * 3  # symbols encode vector of channels into 1 number
        print(f"Image dim: {data[0].shape}={xdim}, latent dim {out.z[0].shape}={zdim}")
        print(f"Final test bits per pixel {bpp}, final mse {self.learner.losses_test['mse'][-1]}")
        stats = {"Bits_per_pixel": bpp, "zdim": zdim, "xdim": xdim}
        wandb.log(stats)
        thetas = self.learner.model.emodel.thetas
        probs = F.softmax(thetas, dim=0)
        centers = list(range(len(thetas)))
        tabledata = [[c, p.item()] for c, p in zip(centers, probs)]
        table = wandb.Table(data=tabledata, columns=["c_idx", "prob(c)"])
        wandb.log({"prob_plot": wandb.plot.bar(table, "c_idx", "prob(c)")})


class SaveModel(Callback):
    """Callback for saving model parameters."""

    def fit_end(self):
        model = self.learner.model
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
