import torch
import torch.nn as nn
from torch.distributions.beta import Beta
from von_mises import *
import probtorch
from utils import global_to_local
import math

class Enc_angle(nn.Module):
    def __init__(self, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.angle_loc = nn.Sequential(
            nn.Linear(2 * D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.angle_log_con = nn.Sequential(
            nn.Linear(2 * D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_loc = torch.zeros(1)
        self.prior_con = torch.ones(1) * 1e-3
        if CUDA:
            with torch.cuda.device(device):
                self.prior_loc = self.prior_loc.cuda()
                self.prior_con = self.prior_con.cuda()
    def forward(self, ob, state, mu):
        q = probtorch.Trace()
        p = probtorch.Trace()
        ob_mu = torch.cat((ob, global_to_local(mu, state)), -1)
        # ob_mu = ob - global_to_local(mu, state)
        q_angle_loc = self.angle_loc(ob_mu)
        q_angle_con = self.angle_log_con(ob_mu).exp()
        q = VonMises(q_angle_loc, q_angle_con)
        p = VonMises(self.prior_loc, self.prior_con)
        angle = q.sample()
        log_q_angle = q.log_prob(angle)
        log_p_angle = p.log_prob(angle)
        # angles = beta_samples
        # beta_samples = angle_samples / (2 * math.pi)
        # beta_samples[beta_samples == 1.0] = 1.0 - 1e-6



        return angle, log_q_angle, log_p_angle
