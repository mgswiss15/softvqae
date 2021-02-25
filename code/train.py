"""

Use configparser for donfiguration: https://hackersandslackers.com/simplify-your-python-projects-configuration/
"""

import torch.optim as optim
import helpers as helpers
from softae import SoftAe, SoftVqAe
from learners import Learner
from torchvision import datasets, transforms
import torch.utils.data as data
import callbacks as cb
import torch
import argparse


def train(config):
    if "cifar" in config.traindir:
        if "small" in config.traindir:
            trainset = helpers.CifarSmall(config.traindir, train=True, transform=transforms.ToTensor())
            testset = helpers.CifarSmall(config.testdir, train=False, transform=transforms.ToTensor())
        else:
            trainset = datasets.CIFAR10(config.traindir, train=True, transform=transforms.ToTensor())
            testset = datasets.CIFAR10(config.testdir, train=False, transform=transforms.ToTensor())
    else:
        trainset = datasets.ImageFolder(config.traindir, transform=transforms.ToTensor())
        testset = datasets.ImageFolder(config.testdir, transform=transforms.ToTensor())
    train_loader = data.DataLoader(trainset, batch_size=config.bsize, shuffle=True)
    test_loader = data.DataLoader(testset, batch_size=config.bsize, shuffle=True)
    model_class = SoftAe if config.novq else SoftVqAe
    model = model_class(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)
    callbacks = [cb.Wandb(), cb.Printouts(), cb.Evaluate(), cb.Scheduler(), cb.SaveModel()]
    learner = Learner(model, optimizer, train_loader, test_loader, config.device, callbacks)
    project_name = "SoftAe" if config.novq else "SoftVqAe"
    helpers.wandb_init(project_name, config)
    learner.fit(config.epochs)
    return model, test_loader


def get_ae_config(cfile):
    parser = helpers.FileParser(fromfile_prefix_chars='@')
    # data
    parser.add_argument("--traindir", type=str, help="Dir with the training data.")
    parser.add_argument("--testdir", type=str, help="Dir with the testing data (only useful if not the same as training).")
    parser.add_argument("--bsize", type=int, help="Batch size for train and test.")
    # sizes
    parser.add_argument("--in_channels", type=int, help="Number of input channels.")
    parser.add_argument("--z_channels", type=int, help="Number of channels in the latent z.")
    # loss
    parser.add_argument("--alpha", type=float, help="Weight of soft entropy loss in total.")
    parser.add_argument("--beta", type=float, help="Weight of entropy loss in total.")
    parser.add_argument("--sigma", type=int, help="Annealing for soft quantization.")
    # ae architecture
    parser.add_argument("--ae_hidden", type=int, help="Number of channels in hidden layers in encoder/decoder.")
    parser.add_argument("--ae_resblocks", type=int, help="Number of resblocks in encoder/decoder.")
    parser.add_argument("--ae_kernel", type=int, help="Size of kernel in down/up-sampling layers.")
    parser.add_argument("--dsf", type=int, choices=[8, 4, 2, 1], help="Downsampling factor (1=no downsampling).")
    parser.add_argument("--novq", action="store_true", help="Do not use vq.")
    # centers
    parser.add_argument("--c_num", type=int, help="Number of centers.")
    parser.add_argument("--c_min", type=float, help="Initial min value of centers.")
    parser.add_argument("--c_max", type=float, help="Initial max value of centers.")
    # training
    parser.add_argument("--random_seed", type=int, default="1234", help="Random seed for RNG for reproducibility.")
    parser.add_argument("--max_lr", type=float, help="Max lr for OneCycleLR scheduler.")
    parser.add_argument("-lr", "--learn_rate", type=float, help="Learning rate - ignored if you use scheduler.")
    parser.add_argument("--epochs", type=int, help="Num of epochs for training.")
    parser.add_argument("--nogpu", action="store_true", help="Do not train on gpu.")
    args = parser.parse_args(['@./configs/ae.ini', '@'+cfile])
    gpu = not args.nogpu
    args.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    args.testdir = args.traindir if args.testdir is None else args.testdir
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfile", type=str, default="./configs/ae.ini",
                        help="Config file for the particular run.")
    args = parser.parse_args()
    config = get_ae_config(args.cfile)
    helpers.reproducibility_init(config.random_seed)
    train(config)
