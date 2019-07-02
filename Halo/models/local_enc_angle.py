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

        self.prior_low = torch.zeros(1)
        self.prior_high = torch.ones(1) * 2 * math.pi
        if CUDA:
            self.prior_low = self.prior_low.cuda().to(device)
            self.prior_high = self.prior_high.cuda().to(device)
    def forward(self, ob, state, mu):
        q = probtorch.Trace()
        p = probtorch.Trace()
        # ob_mu = torch.cat((ob, global_to_local(mu, state)), -1)
        ob_mu = ob - global_to_local(mu, state)
        q_angle_con1 = self.angle_log_con1(ob_mu).exp()
        q_angle_con0 = self.angle_log_con0(ob_mu).exp()
        beta_samples = Beta(q_angle_con1, q_angle_con0).sample()
        angles = beta_samples * 2 * math.pi
        q.beta(q_angle_con1,
               q_angle_con0,
               value=beta_samples,
               name='angles')

        p.uniform(self.prior_low,
                  self.prior_high,
                  value=angles,
                  name='angles')


        return q, p
