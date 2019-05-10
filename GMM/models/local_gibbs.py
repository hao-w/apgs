import torch
import torch.nn as nn
import probtorch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat

class Gibbs_z():
    """
    Gibbs sampling for p(z | mu, tau, x) given mu, tau, x
    """
    def __init__(self, K, CUDA, device):

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)

    def forward(self, obs, obs_tau, obs_mu, N, K):
        q = probtorch.Trace()
        p = probtorch.Trace()
        obs_sigma = 1. / obs_tau.sqrt()
        obs_mu_expand = obs_mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
        obs_sigma_expand = obs_sigma.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
        obs_expand = obs.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
        log_gammas = Normal(obs_mu_expand, obs_sigma_expand).log_prob(obs_expand).sum(-1).transpose(-1, -2) + self.prior_pi.log() # S * B * N * K
        q_probs = F.softmax(log_gammas, dim=-1)
        z = cat(q_probs).sample()
        _ = q.variable(cat, probs=q_probs, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p

    def sample_prior(self, N, sample_size, batch_size):
        p_init_z = cat(self.prior_pi)
        state = p_init_z.sample((sample_size, batch_size, N,))
        return state
