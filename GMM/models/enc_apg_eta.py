import torch
import torch.nn as nn
import probtorch
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from kls_gmm import posterior_eta

class Enc_apg_eta(nn.Module):
    def __init__(self, K, D):
        super(self.__class__, self).__init__()
        self.gamma = nn.Sequential(
            nn.Linear(K+D, K),
            nn.Softmax(-1))
        self.ob = nn.Sequential(
            nn.Linear(K+D, D))

    def forward(self, ob, z, prior_ng, sampled=True, tau_old=None, mu_old=None):
        q = probtorch.Trace()
        (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
        ob_z = torch.cat((ob, z), -1) # concatenate observations and cluster asssignemnts
        q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.ob(ob_z) , self.gamma(ob_z), prior_alpha, prior_beta, prior_mu, prior_nu)
        if sampled == True:
            tau = Gamma(q_alpha, q_beta).sample()
            q.gamma(q_alpha,
                    q_beta,
                    value=tau,
                    name='precisions')
            mu = Normal(q_mu, 1. / (q_nu * q['precisions'].value).sqrt()).sample()
            q.normal(q_mu,
                     1. / (q_nu * q['precisions'].value).sqrt(),
                     value=mu,
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
        return q
