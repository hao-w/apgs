import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
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
            nn.Linear(num_hidden, K*num_stats))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.prior_mean_mu = torch.zeros((K, D))
        self.prior_mean_sigma = torch.ones((K, D)) * 7.0

        if CUDA:
            self.prior_mean_mu = self.prior_mean_mu.cuda().to(device)
            self.prior_mean_sigma = self.prior_mean_sigma.cuda().to(device)

    def forward(self, obs, state, K, sample_size, batch_size):
        q = probtorch.Trace()
        p = probtorch.Trace()
        D = obs.shape[-1]
        neural_stats = self.neural_stats(torch.cat((obs, state), -1)).mean(-2).view(sample_size, batch_size, K, -1) 
        nss1 = torch.cat((neural_stats[:,:,0,:], self.prior_mean_mu[0].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[0].repeat(sample_size, batch_size, 1)), -1)
        nss2 = torch.cat((neural_stats[:,:,1,:], self.prior_mean_mu[1].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[1].repeat(sample_size, batch_size, 1)), -1)
        nss3 = torch.cat((neural_stats[:,:,2,:], self.prior_mean_mu[2].repeat(sample_size, batch_size, 1), self.prior_mean_sigma[2].repeat(sample_size, batch_size, 1)), -1)

        q_mean_mu1 = self.mean_mu(nss1)
        q_mean_sigma1 = self.mean_log_sigma(nss1).exp()

        q_mean_mu2 = self.mean_mu(nss2)
        q_mean_sigma2 = self.mean_log_sigma(nss2).exp()
        
        q_mean_mu3 = self.mean_mu(nss3)
        q_mean_sigma3 = self.mean_log_sigma(nss3).exp()
        
        q_mean_mu = torch.cat((q_mean_mu1.unsqueeze(-2),q_mean_mu2.unsqueeze(-2),q_mean_mu3.unsqueeze(-2)), -2)
        q_mean_sigma = torch.cat((q_mean_sigma1.unsqueeze(-2),q_mean_sigma2.unsqueeze(-2),q_mean_sigma3.unsqueeze(-2)), -2)
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

