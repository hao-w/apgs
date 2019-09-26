import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Encoder(nn.Module):
    def __init__(self, num_pixels, num_hidden1, num_hidden2, num_style, num_digits, CUDA, device):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden1),
                            nn.ReLU())
        self.disc = nn.Linear(num_hidden1, num_digits)
        self.style_mean = nn.Sequential(
                            nn.Linear(num_hidden1 + num_digits, num_hidden2),
                            nn.ReLU(),
                            nn.Linear(num_hidden2, num_style))
        self.style_log_std = nn.Sequential(
                                nn.Linear(num_hidden1 + num_digits, num_hidden2),
                                nn.ReLU(),
                                nn.Linear(num_hidden2, num_style))

        self.prior_pi = torch.ones(num_digits) / .num_digits
        self.prior_mu = torch.zeros(num_style)
        self.prior_std = torch.ones(num_style)

        if CUDA:
            with torch.cuda.device(device):
                self.prior_pi = self.prior_pi.cuda()
                self.prior_mu = self.prior_mu.cuda()
                self.prior_std = self.prior_std.cuda()

    def forward(self, images, labels=None, num_samples=NUM_SAMPLES):
        q = probtorch.Trace()
        p = probtorch.Trace()

        hidden = self.enc_hidden(images)
        q_probs = F.softmax(self.disc(hidden), -1)
        digits = cat(q_probs).sample()
        _ = q.variables(cat,probs=q_probs, value=digits, name='digits')
        _ = p.variables(cat, probs=self.prior_pi, value=digits, name='digits')
        hidden2 = torch.cat([digits, hidden] , -1)
        q_mu = self.style_mean(hidden2)
        q_std = self.style_log_std(hidden2).exp()

        z = Normal(q_mu, q_std).sample()
        q.normal(loc=q_mu,
                 scale=q_std,
                 value=z
                 name='z')
        p.normal(loc=self.prior_mu,
                 scale=self.prior_std,
                 value=z,
                 name='z')
        return q, p
