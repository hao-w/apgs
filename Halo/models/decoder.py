import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import probtorch
import math
from utils import *

class Dec_x(nn.Module):
    def __init__(self, D, num_hidden, CUDA):
        super(self.__class__, self).__init__()
        self.recon_mu = nn.Sequential(
            nn.Linear(D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

    def forward(self, ob, state, mu, angle, recon_sigma):
        p = probtorch.Trace()
        # S, B, N, D = ob.shape
        S, B, N, D = ob.shape
        embedding = torch.cat((global_to_local(mu, state), angle), -1)
        recon_mu = self.recon_mu(embedding)
        p.normal(recon_mu,
                 recon_sigma,
                 value=ob,
                 name='likelihood')

        return p
