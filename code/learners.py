import torch
from collections import defaultdict


class Learner():
    """Class for training VQ-VAE.

    Attributes:
        model (nn.Module): model to be trained
        optimizer (torch.optim.Optimizer): optimizer to be used in gradient descent
        train_loader, test_loader (torch.data.DataLoader): train and test dataloaders
        device (torch.device): for training on cuda or cpu
        callback_list (list): list of callbacks to be executed at various steps in training
        losses_train, losses_test (dict): history (list) of all parts of train/ test losses
    """

    def __init__(self, model, optimizer, train_loader, test_loader, device, callback_list=[]):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.callback_list = callback_list
        self.losses_train = defaultdict(list)
        self.losses_test = defaultdict(list)
        for cb in self.callback_list:
            cb.init_learner(self)

    def fit(self, epochs):
        self.epochs = epochs
        self.callback('fit_begin')
        # self.eval_epoch()
        for self.epoch in range(epochs):
            self.callback('epoch_begin')
            self.train_epoch()
            self.eval_epoch()
            self.callback('epoch_end')
        self.callback('fit_end')

    def train_epoch(self):
        self.model.train()
        for batch in self.train_loader:  # batch is tuple with data and labels
            self.callback('trainbatch_begin')
            batch = batch[0].to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            losses = self.model.loss_func(batch, out)
            self.callback('before_backward')
            losses['total'].backward()
            self.callback('after_backward')
            self.optimizer.step()
            # print("="*50, "Checking crossH computational graph")
            # print(f"losses['crossH']", {losses['crossH'].requires_grad}, {losses['crossH'].grad_fn})
            # thetas = self.model.emodel.thetas
            # print(f"thetas", {thetas.requires_grad}, {thetas.grad_fn}, {thetas.grad})
            # symbols = out.symbols
            # print(f"symbols", {symbols.requires_grad}, {symbols.grad_fn}, {symbols.grad})
            # print("="*50, "Symbols are a leaf not requiring gradient.")
            # assert False
            self.callback('after_optimizer')
            for key, value in losses.items():
                self.losses_train[key].append(value.item())
            self.callback('trainbatch_end')

    def eval_epoch(self):
        losses = {}
        n_samples = 0.
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:  # batch is tuple with data and labels
                batch = batch[0].to(self.device)
                batch_size = batch.shape[0]
                out = self.model(batch)
                losses_tmp = self.model.loss_func(batch, out)
                for key, value in losses_tmp.items():
                    losses[key] = losses.get(key, 0.) + (value.item() * batch_size)
                n_samples += batch_size
        for key, value in losses.items():
            self.losses_test[key].append(value / n_samples)

    def callback(self, cb_name, *args, **kwargs):
        for cb in self.callback_list:
            cb_method = getattr(cb, cb_name, None)
            if cb_method:
                cb_method(*args, **kwargs)
