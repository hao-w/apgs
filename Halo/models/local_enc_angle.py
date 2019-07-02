import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta
import probtorch
import math
from utils import *

class Enc_angle(nn.Module):
    def __init__(self, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.angle_log_con1 = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))
        self.angle_log_con0 = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_con1 = torch.ones(1)
        self.prior_con0 = torch.ones(1)
        if CUDA:
            self.prior_con1 = self.prior_con1.cuda().to(device)
            self.prior_con0 = self.prior_con0.cuda().to(device)
    def forward(self, ob, state, mu):
        q = probtorch.Trace()
        p = probtorch.Trace()
        # ob_mu = torch.cat((ob, global_to_local(mu, state)), -1)
        ob_mu = ob - global_to_local(mu, state)
        q_angle_con1 = self.angle_log_con1(ob_mu).exp()
        q_angle_con0 = self.angle_log_con0(ob_mu).exp()
        beta_samples = Beta(q_angle_con1, q_angle_con0).sample()
        angles = beta_samples
        q.beta(q_angle_con1,
               q_angle_con0,
               value=beta_samples,
               name='angles')

        p.beta(self.prior_con1,
               self.prior_con0,
               value=beta_samples,
               name='angles')


        return q, p
