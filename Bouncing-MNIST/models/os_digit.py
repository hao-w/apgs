import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class OS_digit(nn.Module):
    def __init__(self, num_pixels, num_hidden1, nss_dim, num_hidden2, z_what_dim, CUDA, device):
        super(self.__class__, self).__init__()
        self.enc_nss = nn.Sequential(
                        nn.Linear(num_pixels, num_hidden1),
                        nn.ReLU(),
                        nn.Linear(num_hidden1, nss_dim),
                        nn.ReLU())
        # self.enc_hidden = nn.Sequential(
        #                 nn.Linear(nss_dim, num_hidden1),
        #                 nn.ReLU())
        # self.disc = nn.Sequentialnn.Linear(num_hidden1, num_digits)
        self.z_what_mean = nn.Sequential(
                        nn.Linear(nss_dim, num_hidden2),
                        nn.ReLU(),
                        nn.Linear(num_hidden2, z_what_dim))
        self.z_what_log_std = nn.Sequential(
                        nn.Linear(nss_dim, num_hidden2),
                        nn.ReLU(),
                        nn.Linear(num_hidden2, z_what_dim))
        # self.prior_pi = torch.ones(num_digits) / .num_digits
        self.prior_mu = torch.zeros(z_what_dim)
        self.prior_std = torch.ones(z_what_dim)

        if CUDA:
            with torch.cuda.device(device):
                # self.prior_pi = self.prior_pi.cuda()
                self.prior_mu = self.prior_mu.cuda()
                self.prior_std = self.prior_std.cuda()
    def forward(self, frames, S, sampled=True, z_what_old=None):
        q = probtorch.Trace()
        p = probtorch.Trace()
        B, T, _ = frames.shape
        nss = self.enc_nss(frames.view(B*T, -1)).view(B, T, -1).mean(-1) ## B * nss_dim
        # q_probs = F.softmax(self.disc(hidden), -1)
        # digits = cat(q_probs).sample()
        # _ = q.variables(cat,probs=q_probs, value=digits, name='digits')
        # _ = p.variables(cat, probs=self.prior_pi, value=digits, name='digits')
        # hidden2 = torch.cat([digits, hidden] , -1)
        q_mu = self.style_mean(nss)
        q_std = self.style_log_std(nss).exp()
        z_what = Normal(q_mu, q_std).sample((S,)) ## S * B * z_what_dim
        q.normal(loc=q_mu,
                 scale=q_std,
                 value=z_what,
                 name='z_what')
        p.normal(loc=self.prior_mu,
                 scale=self.prior_std,
                 value=z_what,
                 name='z_what')
        return q, p
