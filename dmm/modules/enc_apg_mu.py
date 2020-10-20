import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import probtorch

class Enc_apg_mu(nn.Module):
    def __init__(self, K, D, num_hidden, num_stats):
        super(self.__class__, self).__init__()
        self.nss1 = nn.Sequential(
            nn.Linear(D+K, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_stats))

        self.nss2 = nn.Sequential(
            nn.Linear(D+K, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K),
            nn.Softmax(-1))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_stats+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

    def forward(self, ob, z, beta, K, priors, sampled=True, mu_old=None, EPS=1e-8):
        q = probtorch.Trace()
        S, B, N, D = ob.shape
        (prior_mu, prior_sigma) = priors
        nss1 = self.nss1(torch.cat((ob, z), -1)).unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2)
        nss2 = self.nss2(torch.cat((ob, z), -1)).unsqueeze(-1).repeat(1, 1, 1, 1, nss1.shape[-1])
        nss = (nss1 * nss2).sum(2) / (nss2.sum(2) + EPS)
        nss_prior = torch.cat((nss, prior_mu.repeat(S, B, K, 1), prior_sigma.repeat(S, B, K, 1)), -1)
        q_mu_mu= self.mean_mu(nss_prior)
        q_mu_sigma = self.mean_log_sigma(nss_prior).exp()
        if sampled:
            mu = Normal(q_mu_mu, q_mu_sigma).sample()
            q.normal(q_mu_mu,
                     q_mu_sigma,
                     value=mu,
                     name='means')
        else:
            q.normal(q_mu_mu,
                     q_mu_sigma,
                     value=mu_old,
                     name='means')

        return q
