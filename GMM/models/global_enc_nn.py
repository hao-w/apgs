import torch
import torch.nn as nn
import probtorch
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from normal_gamma import *

class Enc_eta(nn.Module):
    def __init__(self, K, D, num_hidden, CUDA, device, Reparameterized):
        super(self.__class__, self).__init__()

        self.Reparameterized = Reparameterized
        self.ob = nn.Sequential(
            nn.Linear(K+D, D))
        
        self.tau_log_alpha = nn.Sequential(
            nn.Linear(2*D+1+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))
        
        self.tau_log_beta = nn.Sequential(
            nn.Linear(2*D+1+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))
        
        self.mu_mu = nn.Sequential(
            nn.Linear(2*D+1+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mu_log_nu = nn.Sequential(
            nn.Linear(2*D+1+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))
        
        self.prior_mu = torch.zeros((K, D))
        self.prior_nu = torch.ones((K, D)) * 0.3
        self.prior_alpha = torch.ones((K, D)) * 4
        self.prior_beta = torch.ones((K, D)) * 4
        if CUDA:
            self.prior_mu = self.prior_mu.cuda().to(device)
            self.prior_nu = self.prior_nu.cuda().to(device)
            self.prior_alpha = self.prior_alpha.cuda().to(device)
            self.prior_beta = self.prior_beta.cuda().to(device)

    def forward(self, obs, state, K, D):
        q = probtorch.Trace()
        p = probtorch.Trace()
        local_vars = torch.cat((obs, state), -1)
        xs = self.ob(local_vars)  # S * B * N * D --> S * B * N * D
        stat1, stat2, stat3 = data_to_stats(xs, state, K, D)
        nat1, nat2, nat3, nat4 = params_to_nats(self.prior_alpha[0], self.prior_beta[0], self.prior_mu[0], self.prior_nu[0])
        stat1 = stat1.unsqueeze(-1)
        ss1 = torch.cat((stat1[:,:,0,:], stat2[:,:,0,:], stat3[:,:,0,:]), -1)
        ss2 = torch.cat((stat1[:,:,1,:], stat2[:,:,1,:], stat3[:,:,1,:]), -1)
        ss3 = torch.cat((stat1[:,:,2,:], stat2[:,:,2,:], stat3[:,:,2,:]), -1)
        
        q_nat1
        q_alpha, q_beta, q_mu, q_nu = Post_eta(xs, gammas,
                                                 self.prior_alpha, self.prior_beta, self.prior_mu, self.prior_nu, K, D)
        if self.Reparameterized:
            q.gamma(q_alpha,
                    q_beta,
                    name='precisions')
            q.normal(q_mu,
                     1. / (q_nu * q['precisions'].value).sqrt(),
                     name='means')
        else:
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
        return obs_mu, obs_tau
