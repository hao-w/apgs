import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Enc_digit(nn.Module):
    def __init__(self, num_pixels, num_hidden, z_what_dim, CUDA, device):
        super(self.__class__, self).__init__()
        self.enc_nss = nn.Sequential(
                        nn.Linear(num_pixels, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, int(0.5*num_hidden)),
                        nn.ReLU())
        self.q_mean = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), int(0.25*num_hidden)),
                        nn.ReLU(),
                        nn.Linear(int(0.25*num_hidden), z_what_dim))
        self.q_log_std = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), int(0.25*num_hidden)),
                        nn.ReLU(),
                        nn.Linear(int(0.25*num_hidden), z_what_dim))
        self.prior_mu = torch.zeros(z_what_dim)
        self.prior_std = torch.ones(z_what_dim)

        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_std = self.prior_std.cuda()
    def forward(self, frames, z_where, crop, sampled=True, z_what_old=None):
        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, T, _ = z_where.shape
        nss = self.enc_nss(crop.frame_to_digit(frames, z_where).view(S, B, T, 28*28)).mean(2) ## S * B * nss_dim

        q_mu = self.q_mean(nss)
        q_std = self.q_log_std(nss).exp()
        if sampled:
            z_what = Normal(q_mu, q_std).sample() ## S * B * z_what_dim
            q.normal(loc=q_mu,
                     scale=q_std,
                     value=z_what,
                     name='z_what')
            p.normal(loc=self.prior_mu,
                     scale=self.prior_std,
                     value=z_what,
                     name='z_what')
        else:
            q.normal(loc=q_mu,
                     scale=q_std,
                     value=z_what_old,
                     name='z_what')
            p.normal(loc=self.prior_mu,
                     scale=self.prior_std,
                     value=z_what_old,
                     name='z_what')
        return q, p
