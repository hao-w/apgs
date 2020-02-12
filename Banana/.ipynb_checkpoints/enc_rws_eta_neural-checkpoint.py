import torch
import torch.nn as nn
import probtorch
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma

class Enc_rws_eta(nn.Module):
    def __init__(self, K, D, num_stats, num_hidden):
        super(self.__class__, self).__init__()

        self.nss = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))

        self.tau_alpha = nn.Sequential(
            nn.Linear(D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.tau_beta = nn.Sequential(
            nn.Linear(3*D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mu_mu = nn.Sequential(
            nn.Linear(3*D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mu_nu = nn.Sequential(
            nn.Linear(D+1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

    def forward(self, ob, prior_ng, sampled=True, tau_old=None, mu_old=None):
        q = probtorch.Trace()
        (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
        S, B, N, D = ob.shape
        nss = self.nss(ob).mean(-2)
        q_alpha = self.tau_alpha(nss).exp()
        q_beta = self.tau_beta(nss).exp()
        q_mu = self.mu_mu(nss)
        q_nu = self.mu_nu(nss).exp()
        if sampled:
            tau = Gamma(q_alpha, q_beta).sample()
            q.gamma(q_alpha,
                    q_beta,
                    value=tau,
                    name='precisions')
            mu = Normal(q_mu, 1. / (q_nu * q['precisions'].value).sqrt()).sample()
            q.normal(q_mu,
                     1. / (q_nu * q['precisions'].value).sqrt(), # std = 1 / sqrt(nu * tau)
                     value=mu,
                     name='means')
        else: ## used in backward transition kernel where samples were given from last sweep
            q.gamma(q_alpha,
                    q_beta,
                    value=tau_old,
                    name='precisions')
            q.normal(q_mu,
                     1. / (q_nu * q['precisions'].value).sqrt(),
                     value=mu_old,
                     name='means')
        return q
