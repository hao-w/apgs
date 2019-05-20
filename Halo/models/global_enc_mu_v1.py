import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Enc_mu(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device, Reparameterized):
        super(self.__class__, self).__init__()
        self.Reparameterized = Reparameterized

        self.neural_stats = nn.Sequential(
            nn.Linear(K+D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.prior_mean_mu = torch.zeros((K, D))
        self.prior_mean_sigma = torch.ones((K, D)) * 5.0

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)

    def forward(self, obs, state, K, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()

        neural_stats = self.neural_stats(torch.cat((obs, state), -1))
        _, _, _, stat_size = neural_stats.shape
        cluster_size = state.sum(-2)
        cluster_size[cluster_size == 0.0] = 1.0 # S * B * K
        neural_stats_expand = neural_stats.unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2) ## S * B * N * K * STAT_SIZE
        states_expand = state.unsqueeze(-1).repeat(1, 1, 1, 1, stat_size) ## S * B * N * K * STAT_SIZE
        sum_stats = (states_expand * neural_stats_expand).sum(2) ## S * B * K * STAT_SIZE
        mean_stats = sum_stats / cluster_size.unsqueeze(-1)
        mus = []
        sigmas = []
        for k in range(K):
            stat_muk = torch.cat((self.prior_mean_mu[k].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[k].repeat(sample_size, batch_size, 1), mean_stats[:,:,k,:]), -1)
            mu_k = self.mean_mu(stat_muk)
            sigma_k = self.mean_log_sigma(stat_muk).exp()
            mus.append(mu_k.unsqueeze(-2))
            sigmas.append(sigma_k.unsqueeze(-2))

        q_mean_mu = torch.cat(mus, -2)
        q_mean_sigma = torch.cat(sigmas, -2)
        if self.Reparameterized:
            q.normal(q_mean_mu,
                     q_mean_sigma,
                     name='means')
        else:
            means = Normal(q_mean_mu, q_mean_sigma).sample()
            q.normal(q_mean_mu,
                     q_mean_sigma,
                     value=means,
                     name='means')

        p.normal(self.prior_mean_mu,
                 self.prior_mean_sigma,
                 value=q['means'],
                 name='means')
        return q, p

    def sample_prior(self, sample_size, batch_size):
        p_mu = Normal(self.prior_mean_mu, self.prior_mean_sigma)
        obs_mu = p_mu.sample((sample_size, batch_size,))
        return obs_mu
