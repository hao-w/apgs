import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.beta import Beta
import probtorch

class Enc_apg_local(nn.Module):
    def __init__(self, K, D, num_hidden_state, num_hidden_angle):
        super(self.__class__, self).__init__()

        self.pi_log_prob = nn.Sequential(
            nn.Linear(D, num_hidden_state),
            nn.ReLU(),
            nn.Linear(num_hidden_state, 1))

        self.angle_log_con1 = nn.Sequential(
            nn.Linear(D, num_hidden_angle),
            nn.Tanh(),
            nn.Linear(num_hidden_angle, 1))

        self.angle_log_con0 = nn.Sequential(
            nn.Linear(D, num_hidden_angle),
            nn.Tanh(),
            nn.Linear(num_hidden_angle, 1))

    def forward(self, ob, mu, K, sampled=True, z_old=None, beta_old=None):
        q = probtorch.Trace()
        S, B, N, D = ob.shape
        ob_mu = ob.unsqueeze(2).repeat(1, 1, K, 1, 1) - mu.unsqueeze(-2).repeat(1, 1, 1, N, 1)
        q_probs = F.softmax(self.pi_log_prob(ob_mu).squeeze(-1).transpose(-1, -2), -1)
        if sampled:
            z = cat(q_probs).sample()
            _ = q.variable(cat,
                           probs=q_probs,
                           value=z,
                           name='states')
            mu_expand = torch.gather(mu, -2, z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, D))
            q_angle_con1 = self.angle_log_con1(ob - mu_expand).exp()
            q_angle_con0 = self.angle_log_con0(ob - mu_expand).exp()
            beta = Beta(q_angle_con1, q_angle_con0).sample()
            q.beta(q_angle_con1,
                   q_angle_con0,
                   value=beta,
                   name='angles')
        else:
            _ = q.variable(cat,
                           probs=q_probs,
                           value=z_old,
                           name='states')
            mu_expand = torch.gather(mu, -2, z_old.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, D))
            q_angle_con1 = self.angle_log_con1(ob - mu_expand).exp()
            q_angle_con0 = self.angle_log_con0(ob - mu_expand).exp()
            q.beta(q_angle_con1,
                   q_angle_con0,
                   value=beta_old,
                   name='angles')
        return q
