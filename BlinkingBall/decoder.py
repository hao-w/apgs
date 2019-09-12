import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import probtorch
import math

class Dec_x(nn.Module):
    def __init__(self, D, num_hidden, num_pixels):
        super(self.__class__, self).__init__()
        self.recon_mu = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 2*num_hidden),
            nn.ReLU(),
            nn.Linear(2*num_hidden, num_pixels),
            nn.Sigmoid())

    def forward(self, ob, mu):
        p = probtorch.Trace()
        recon_mu = self.recon_mu(mu)
        p.loss(binary_cross_entropy, recon_mu, ob, name='images')
        return p

def binary_cross_entropy(x_mean, x, EPS=1e-9):
    return - (torch.log(x_mean + EPS) * x + 
              torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)