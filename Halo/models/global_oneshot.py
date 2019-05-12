import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Oneshot_mu(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device, Reparameterized):
        super(self.__class__, self).__init__()

        self.Reparameterized = Reparameterized

        self.neural_stats = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats+2*K*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K*D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats+2*K*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K*D))

        self.prior_mean_mu = torch.zeros(K*D)
        self.prior_mean_sigma = torch.ones(K*D) * 5.0

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)

    def forward(self, obs, K, D, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()

        neural_stats = self.neural_stats(obs)
        mean_stats = neural_stats.mean(-2)  # S * B * STAT_DIM

        stat_mu = torch.cat((self.prior_mean_mu.repeat(sample_size, batch_size, 1), self.prior_mean_sigma.repeat(sample_size, batch_size, 1), mean_stats), -1)

        q_mean_mu = self.mean_mu(stat_mu).view(sample_size, batch_size, K, D)
        q_mean_sigma = self.mean_log_sigma(stat_mu).exp().view(sample_size, batch_size, K, D)

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

        p.normal(self.prior_mean_mu.view(K, D),
                 self.prior_mean_sigma.view(K, D),
                 value=q['means'],
                 name='means')
        return q, p
