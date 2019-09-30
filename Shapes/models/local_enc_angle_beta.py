import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import probtorch
from utils import global_to_local
import math

class Enc_angle(nn.Module):
    def __init__(self, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.angle_mu = nn.Sequential(
            nn.Linear(2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.angle_log_std = nn.Sequential(
            nn.Linear(2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.prior_mu = torch.zeros(D)
        self.prior_std = torch.ones(D)
        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_std = self.prior_std.cuda()
    def forward(self, ob, state, mu, sampled=True, beta_old=None):
        q = probtorch.Trace()
        p = probtorch.Trace()
        # ob_mu = torch.cat((ob, global_to_local(mu, state)), -1)
        ob_mu = torch.cat((ob, global_to_local(mu, state)), -1)
        q_mu = self.angle_mu(ob_mu)
        q_std = self.angle_log_std(ob_mu).exp()
        if sampled == True:
            beta = Normal(q_mu, q_std).sample()
            q.normal(q_mu,
                   q_std,
                   value=beta,
                   name='angles')

            p.normal(self.prior_mu,
                   self.prior_std,
                   value=beta,
                   name='angles')
        else:
            q.normal(q_mu,
                   q_std,
                   value=beta_old,
                   name='angles')

            p.normal(self.prior_mu,
                   self.prior_std,
                   value=beta_old,
                   name='angles')
        return q, p
