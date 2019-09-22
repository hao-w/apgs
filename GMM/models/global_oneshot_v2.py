import torch
import torch.nn as nn
import probtorch
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from normal_gamma import *

class Oneshot_eta(nn.Module):
    def __init__(self, K, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()

        self.gamma = nn.Sequential(
            nn.Linear(D, K),
            nn.Softmax(-1))

        self.ob = nn.Sequential(
            nn.Linear(D, D))

        self.prior_mu = torch.zeros((K, D))
        self.prior_nu = torch.ones((K, D)) * 0.3
        self.prior_alpha = torch.ones((K, D)) * 4
        self.prior_beta = torch.ones((K, D)) * 4
        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_nu = self.prior_nu.cuda()
                self.prior_alpha = self.prior_alpha.cuda()
                self.prior_beta = self.prior_beta.cuda()

        self.q_alpha = nn.Sequential(
            nn.Linear(5+2, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.q_nu = nn.Sequential(
            nn.Linear(5+2, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.q_mu = nn.Sequential(
            nn.Linear(5+2+2, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.q_beta = nn.Sequential(
            nn.Linear(5+2+2, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

    def forward(self, ob, sampled=True, tau_old=None, mu_old=None):
        S, B, K, D = ob.shape
        q = probtorch.Trace()
        p = probtorch.Trace()
        gammas = self.gamma(ob) # S * B * N * K --> S * B * N * K
        xs = self.ob(ob)  # S * B * N * D --> S * B * N * D
        stat1, stat2, stat3 = data_to_stats(xs, gammas)
        nss = torch.cat((stat1.unsqueeze(-1), stat2, stat3), -1)
        q_alpha = self.q_alpha(torch.cat((nss, self.prior_alpha.repeat(S, B, 1, 1)), -1)).exp()
        q_nu = self.q_nu(torch.cat((nss, self.prior_nu.repeat(S, B, 1, 1)), -1)).exp()
        q_mu = self.q_mu(torch.cat((nss, self.prior_mu.repeat(S, B, 1, 1), self.prior_nu.repeat(S, B, 1, 1)), -1))
        q_beta = self.q_beta(torch.cat((nss, self.prior_beta.repeat(S, B, 1, 1), self.prior_nu.repeat(S, B, 1, 1)), -1)).exp()
        # q_alpha, q_beta, q_mu, q_nu = Post_eta(xs, gammas,
        #                                          self.prior_alpha, self.prior_beta, self.prior_mu, self.prior_nu)

        if sampled:
            ## sample in non-reparameterized way
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
        else:
            q.gamma(q_alpha,
                    q_beta,
                    value=tau_old,
                    name='precisions')
            q.normal(q_mu,
                     1. / (q_nu * q['precisions'].value).sqrt(),
                     value=mu_old,
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
