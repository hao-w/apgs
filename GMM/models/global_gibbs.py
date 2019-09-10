import torch
import torch.nn as nn
import probtorch
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from normal_gamma import *

class Gibbs_eta(nn.Module):
    def __init__(self, K, D, CUDA, device):
        super(self.__class__, self).__init__()


        self.prior_mu = torch.zeros((K, D))
        self.prior_nu = torch.ones((K, D)) * 0.3
        self.prior_alpha = torch.ones((K, D)) * 2
        self.prior_beta = torch.ones((K, D)) * 2
        if CUDA:
            self.prior_mu = self.prior_mu.cuda().to(device)
            self.prior_nu = self.prior_nu.cuda().to(device)
            self.prior_alpha = self.prior_alpha.cuda().to(device)
            self.prior_beta = self.prior_beta.cuda().to(device)

    def forward(self, obs, state, K, D):
        q = probtorch.Trace()
        p = probtorch.Trace()

        q_alpha, q_beta, q_mu, q_nu = Post_eta(obs, state,
                                                 self.prior_alpha, self.prior_beta, self.prior_mu, self.prior_nu, K, D)

        precisions = Gamma(q_alpha, q_beta).sample()
        q.gamma(q_alpha,
                q_beta,
                value=precisions,
                name='precisions')

        means = Normal(q_mu, 1. / (q_nu * q['precisions'].value).sqrt()).sample()
        q.normal(q_mu,
                 1. / (q_nu * q['precisions'].value).sqrt(),
                 value=means,
                 name='means')
        ## prior distributions
        p.gamma(self.prior_alpha,
                self.prior_beta,
                value=q['precisions'],
                name='precisions')
        p.normal(self.prior_mu,
                 1. / (self.prior_nu * p['precisions'].value).sqrt(),
                 value=q['means'],
                 name='means')
        return q, p, q_nu

    def sample_prior(self, sample_size, batch_size):
        p_tau = Gamma(self.prior_alpha, self.prior_beta)
        obs_tau = p_tau.sample((sample_size, batch_size,))
        p_mu = Normal(self.prior_mu.repeat(sample_size, batch_size, 1, 1), 1. / (self.prior_nu * obs_tau).sqrt())
        obs_mu = p_mu.sample()
        log_p = p_mu.log_prob(obs_mu) + p_tau.log_prob(obs_tau)
        return obs_tau, obs_mu, log_p
