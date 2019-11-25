import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import probtorch

class Enc_digit(nn.Module):
    def __init__(self, num_pixels, num_hidden, z_what_dim):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
                        nn.Linear(num_pixels, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, int(0.5*num_hidden)),
                        nn.ReLU())
        self.q_mean = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), z_what_dim))
        self.q_log_std = nn.Sequential(
                        nn.Linear(int(0.5*num_hidden), z_what_dim))

    def forward(self, cropped, sampled=True, z_what_old=None):
        q = probtorch.Trace()
        hidden = self.enc_hidden(cropped)
        q_mu = self.q_mean(hidden).mean(2) ## because T is on the 3rd dim in cropped
        q_std = self.q_log_std(hidden).exp().mean(2)
        if sampled:
            z_what = Normal(q_mu, q_std).rsample() ## S * B * K * z_what_dim
            q.normal(loc=q_mu,
                     scale=q_std,
                     value=z_what,
                     name='z_what')
        else:
            q.normal(loc=q_mu,
                     scale=q_std,
                     value=z_what_old,
                     name='z_what')

        return q
