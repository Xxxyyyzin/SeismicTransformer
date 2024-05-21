# DISCLAIMER: This script is borrowed from Moseley (2018-2020).
# Source: [https://github.com/benmoseley/seismic-simulation-complex-media]

import math

import numpy as np
import torch

# # #ImportError: cannot import name 'Constants'
# from constants_AutoEncoder import Constants
# c=Constants()


# Note: trainer passes arguments in the following order: loss(*labels, *outputs, constants)

def l2_mean_loss(a, b, c=None):
    return torch.mean((a - b) ** 2)


def l2_sum_loss(a, b, c=None):
    return torch.sum((a - b) ** 2)


def l1_mean_loss(a, b, c=None):
    return torch.mean(torch.abs(a - b))


def l1_sum_loss(a, b, c=None):
    return torch.sum(torch.abs(a - b))


def l1_mean_loss_gain(a, b, c=None):
    # apply loss with ^T_GAIN gain profile
    device = torch.device("cuda:%i" % (c.DEVICE) if torch.cuda.is_available() else "cpu")

    t_gain = np.arange(512, dtype=np.float32) **c.T_GAIN
    t_gain = t_gain / np.median(t_gain)
    t_gain = t_gain.reshape((1, 1, 1, 512))  # along NSTEPS
    t_gain = torch.from_numpy(t_gain).to(device)
    # t_gain = torch.from_numpy(t_gain)
    g = t_gain * (a - b)
    return torch.mean(torch.abs(g))

def l1_half_loss(a,b):
    a=l1_mean_loss(a,b)
    return torch.sqrt(a)


def l2_mean_loss_gain(a, b, c=None):
    # apply loss with ^T_GAIN gain profile
    device = torch.device("cuda:%i" % (c.DEVICE) if torch.cuda.is_available() else "cpu")

    t_gain = np.arange(512, dtype=np.float32) ** 2.5
    t_gain = t_gain / np.median(t_gain)
    t_gain = t_gain.reshape((1, 1, 1, 512))  # along NSTEPS
    t_gain = torch.from_numpy(t_gain).to(device)
    g = t_gain * (a - b)
    return torch.mean(g ** 2)


if __name__ == "__main__":
    a = torch.ones((10, 20))
    b = torch.ones((10, 20)).mul_(0.1)  # (in place)

    print(l2_sum_loss(a, b))

    print(l2_mean_loss(a, b))
    print(l1_mean_loss(a, b))
