import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import probtorch
import math
from utils import *

class Dec_x(nn.Module):
    def __init__(self, K, D, num_hidden, recon_sigma, CUDA, device):
        super(self.__class__, self).__init__()
        self.recon_mu = nn.Sequential(
            nn.Linear(1+K+D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))
        self.recon_log_std = nn.Sequential(
            nn.Linear(1+K+D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))
        # self.recon_sigma = recon_sigma
        # self.radi = torch.ones(1)
        # if CUDA:
        #     with torch.cuda.device(device):
                # self.recon_sigma = self.recon_sigma.cuda()
                # self.radi = self.radi.cuda()
        # self.recon_sigma = nn.Parameter(self.recon_sigma)
        # self.radi = nn.Parameter(self.radi)

    def forward(self, ob, state, angle, mu):
        p = probtorch.Trace()
        S, B, N, D = ob.shape
        embed = torch.cat((angle, state, global_to_local(mu, state)),-1)
        q_centered = self.recon_mu(embed)
        q_std = self.recon_log_std(embed).exp()
        #  = a / (a**2).sum(-1).unsqueeze(-1).sqrt()
        recon_mu = q_centered + global_to_local(mu, state)
        p.normal(recon_mu,
                 q_std,
                 value=ob,
                 name='likelihood')

        return p
