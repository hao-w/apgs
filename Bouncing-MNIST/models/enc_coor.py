import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.distributions.normal import Normal
import probtorch
import math

class Enc_coor(nn.Module):
    def __init__(self, D, num_pixels, num_hidden):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
        self.where_mean = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), D),
                            nn.Tanh())

        self.where_log_std = nn.Sequential(
                            nn.Linear(num_hidden, int(0.5*num_hidden)),
                            nn.ReLU(),
                            nn.Linear(int(0.5*num_hidden), D))
        #
        # self.enc = nn.Sequential(
        #                     nn.Linear(num_pixels, num_hidden),
        #                     nn.ReLU(),
        #                     nn.Linear(num_hidden, int(0.5*num_hidden)),
        #                     nn.ReLU(),
        #                     nn.Linear(int(0.5*num_hidden), D),
        #                     nn.Tanh())

    def forward(self, conved, sampled=True, z_where_old=None):
        q = probtorch.Trace()
        hidden = self.enc_hidden(conved)
        q_mean = self.where_mean(hidden)
        q_std = self.where_log_std(hidden).exp()
        # if z_where_t_1 i None:
        if sampled:
            z_where = Normal(q_mean, q_std).sample()
            q.normal(loc=q_mean, scale=q_std, value=z_where, name='z_where')
        else:
            q.normal(loc=q_mean, scale=q_std, value=z_where_old, name='z_where')
        return q
