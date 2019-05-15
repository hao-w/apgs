import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Enc_eta(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device, Reparameterized):
        super(self.__class__, self).__init__()
        self.Reparameterized = Reparameterized

        self.neural_stats = nn.Sequential(
            nn.Linear(K+D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.radi_log_alpha = nn.Sequential(
            nn.Linear(num_stats+D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.radi_log_beta = nn.Sequential(
            nn.Linear(num_stats+D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_mean_mu = torch.zeros((K, D))
        self.prior_mean_sigma = torch.ones((K, D)) * 5.0
        self.prior_radi_alpha = torch.ones((K, 1))
        self.prior_radi_beta = torch.ones((K, 1))

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)
            self.prior_radi_alpha = torch.ones((K, 1)).cuda().to(device)
            self.prior_radi_beta = torch.ones((K, 1)).cuda().to(device)

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

        q_mean_mu = torch.cat((self.mean_mu(mean_stats[:,:,0,:]).unsqueeze(-2), self.mean_mu(mean_stats[:,:,1,:]).unsqueeze(-2), self.mean_mu(mean_stats[:,:,2,:]).unsqueeze(-2)), -2)
        q_mean_sigma = torch.cat((self.mean_log_sigma(mean_stats[:,:,0,:]).exp().unsqueeze(-2), self.mean_log_sigma(mean_stats[:,:,1,:]).exp().unsqueeze(-2), self.mean_log_sigma(mean_stats[:,:,2,:]).exp().unsqueeze(-2)), -2)

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

        means = q['means']
        stat_mu1 = torch.cat((mean_stats[:,:,0,:].unsqueeze(-1), means[:,:,0,:].unsqueeze(-1), -1))
        stat_mu2 = torch.cat((mean_stats[:,:,1,:].unsqueeze(-1), means[:,:,1,:].unsqueeze(-1), -1))
        stat_mu3 = torch.cat((mean_stats[:,:,2,:].unsqueeze(-1), means[:,:,2,:].unsqueeze(-1), -1))

        q_radi_alpha = torch.cat((self.radi_log_alpha(stat_mu1).exp().unsqueeze(-2), self.radi_log_alpha(stat_mu2).exp().unsqueeze(-2), self.radi_log_alpha(stat_mu3).exp().unsqueeze(-2)), -2)
        q_radi_beta = torch.cat((self.radi_log_beta(stat_mu1).exp().unsqueeze(-2), self.radi_log_beta(stat_mu2).exp().unsqueeze(-2), self.radi_log_beta(stat_mu3).exp().unsqueeze(-2)), -2)

        if self.Reparameterized:
            q.gamma(q_radi_alpha,
                    q_radi_beta,
                    name='radi')
        else:
            radis = Gamma(q_radi_alpha, q_radi_beta).sample()
            q.gamma(q_radi_alpha,
                    q_radi_beta,
                    value=radis,
                    name='radi')

        p.gamma(self.prior_radi_alpha,
                self.prior_radi_beta,
                value=q['radi'],
                name='radi')
        return q, p

    def sample_prior(self, sample_size, batch_size):
        p_mu = Normal(self.prior_mean_mu, self.prior_mean_sigma)
        obs_mu = p_mu.sample((sample_size, batch_size,))
        return obs_mu
