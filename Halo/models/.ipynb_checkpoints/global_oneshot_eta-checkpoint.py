import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Oneshot_eta(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats, CUDA, device, Reparameterized):
        super(self.__class__, self).__init__()

        self.Reparameterized = Reparameterized

        self.neural_stats = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.radi_log_alpha = nn.Sequential(
            nn.Linear(num_stats, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K))

        self.radi_log_beta = nn.Sequential(
            nn.Linear(num_stats, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K))

        self.prior_mean_mu = torch.zeros((K, D))
        self.prior_mean_sigma = torch.ones((K, D)) * 7.0
        self.prior_radi_alpha = torch.ones((K, 1)) * 8
        self.prior_radi_beta = torch.ones((K, 1)) * 16

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)
            self.prior_radi_alpha = torch.ones((K, 1)).cuda().to(device)
            self.prior_radi_beta = torch.ones((K, 1)).cuda().to(device)

    def forward(self, obs, K):
        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, N, D = obs.shape

        neural_stats = self.neural_stats(obs)
        mean_stats = neural_stats.mean(-2)  # S * B * STAT_DIM

        q_radi_alpha = self.radi_log_alpha(mean_stats).exp().unsqueeze(-1)
        q_radi_beta = self.radi_log_beta(mean_stats).exp().unsqueeze(-1)

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
        
        radi = q['radi'].value
        
        stat_radi1 = torch.cat((mean_stats, radi[:,:,0, :]), -1)
        stat_radi2 = torch.cat((mean_stats, radi[:,:,1, :]), -1)
        stat_radi3 = torch.cat((mean_stats, radi[:,:,2, :]), -1)        

        q_mean_mu = torch.cat((self.mean_mu(stat_radi1).unsqueeze(-2), self.mean_mu(stat_radi2).unsqueeze(-2), self.mean_mu(stat_radi3).unsqueeze(-2)), -2)
        q_mean_sigma = torch.cat((self.mean_log_sigma(stat_radi1).exp().unsqueeze(-2), self.mean_log_sigma(stat_radi2).exp().unsqueeze(-2), self.mean_log_sigma(stat_radi3).exp().unsqueeze(-2)), -2)

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
