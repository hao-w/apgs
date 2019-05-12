import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Gibbs_z():
    """
    Gibbs sampling for p(z | mu, tau, x) given mu, tau, x
    """
    def __init__(self, K, CUDA, device):

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)

    def forward(self, obs, obs_mu, obs_rad, noise_sigma, N, K, sample_size, batch_size):
        obs_mu_expand = obs_mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
        obs_expand = obs.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
        distance = ((obs_expand - obs_mu_expand)**2).sum(-1).sqrt()
        obs_dist = Normal(obs_rad,  noise_sigma)
        log_distance = (obs_dist.log_prob(distance) - (2*math.pi*distance).log()).transpose(-1, -2) + self.prior_pi.log() # S * B * N * K

        q_pi = F.softmax(log_distance, -1)
        q = probtorch.Trace()
        p = probtorch.Trace()
        z = cat(q_pi).sample()
        _ = q.variable(cat, probs=q_pi, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p

    def sample_prior(self, N, sample_size, batch_size):
        p_init_z = cat(self.prior_pi)
        state = p_init_z.sample((sample_size, batch_size, N,))
        return state
