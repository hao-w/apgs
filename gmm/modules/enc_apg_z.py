import torch
import torch.nn as nn
import torch.nn.functional as F
import probtorch
from torch.distributions.one_hot_categorical import OneHotCategorical as cat

class Enc_apg_z(nn.Module):
    def __init__(self, K, D, num_hidden):
        super(self.__class__, self).__init__()
        self.pi_log_prob = nn.Sequential(
            nn.Linear(3*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

    def forward(self, ob, tau, mu, sampled=True, z_old=None):
        q = probtorch.Trace()
        gamma_list = []
        N = ob.shape[-2]
        for k in range(mu.shape[-2]):
            data_ck = torch.cat((ob, mu[:, :, k, :].unsqueeze(-2).repeat(1,1,N,1), tau[:, :, k, :].unsqueeze(-2).repeat(1, 1, N, 1)), -1) ## S * B * N * 3D
            gamma_list.append(self.pi_log_prob(data_ck))
        q_probs = F.softmax(torch.cat(gamma_list, -1), -1)
        if sampled == True:
            z = cat(q_probs).sample()
            _ = q.variable(cat, probs=q_probs, value=z, name='states')
        else:
            _ = q.variable(cat, probs=q_probs, value=z_old, name='states')
        return q
