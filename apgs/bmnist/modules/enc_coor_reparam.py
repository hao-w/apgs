import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import probtorch

class Enc_coor(nn.Module):
    def __init__(self, num_pixels, num_hidden, z_where_dim):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
        self.where_mean = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), z_where_dim),
                            nn.Tanh())

        self.where_log_std = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.Softplus(),
                            nn.Linear(int(0.5*num_hidden), z_where_dim))


    def forward(self, conved, sampled=True, z_where_old=None):
        q = probtorch.Trace()
        hidden = self.enc_hidden(conved)
        q_mean = self.where_mean(hidden)
        q_std = self.where_log_std(hidden).exp()
        if sampled:
            z_where = Normal(q_mean, q_std).rsample()
            q.normal(loc=q_mean, scale=q_std, value=z_where, name='z_where')
        else:
            q.normal(loc=q_mean, scale=q_std, value=z_where_old, name='z_where')
        return q
