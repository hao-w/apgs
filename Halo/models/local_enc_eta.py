import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Enc_z(nn.Module):
    def __init__(self, K, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.pi_log_prob = nn.Sequential(
            nn.Linear(2*D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)

    def forward(self, obs, obs_mu, radi, noise_sigma):
        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, N, D = obs.shape
        K = obs_mu.shape[-2]
        gamma_list = []
        for k in range(K):
            data_ck = torch.cat((obs, obs_mu[:, :, k, :].unsqueeze(-2).repeat(1,1,N,1), radi[:, :, k, :].unsqueeze(-2).repeat(1,1,N,1)), -1)
            gamma_list.append(self.pi_log_prob(data_ck))
        q_probs = F.softmax(torch.cat(gamma_list, -1), -1) # S * B * N * K
        z = cat(q_probs).sample()
        _ = q.variable(cat, probs=q_probs, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p
    
    def sample_prior(self, N, sample_size, batch_size):
        p_init_z = cat(self.prior_pi)
        state = p_init_z.sample((sample_size, batch_size, N,))
        return state