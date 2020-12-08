import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from apgs.gmm.kls_gmm import posterior_eta
import probtorch

class Enc_rws_eta(nn.Module):
    """
    One-shot (i.e.RWS) encoder of {mean, covariance} i.e. \eta
    """
    def __init__(self, K, D):
        super(self.__class__, self).__init__()

        self.rws_eta_gamma = nn.Sequential(
            nn.Linear(D, K),
            nn.Softmax(-1))

        self.rws_eta_ob = nn.Sequential(
            nn.Linear(D, D))

    def forward(self, ob, prior_ng, sampled=True, tau_old=None, mu_old=None):
        q = probtorch.Trace()
        (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
        q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.rws_eta_ob(ob) , self.rws_eta_gamma(ob), prior_alpha, prior_beta, prior_mu, prior_nu)
        if sampled: ## used in forward transition kernel where we need to sample
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
    
class Enc_apg_eta(nn.Module):
    """
    Conditional proposal of {mean, covariance} i.e. \eta
    """
    def __init__(self, K, D):
        super(self.__class__, self).__init__()
        self.apg_eta_gamma = nn.Sequential(
            nn.Linear(K+D, K),
            nn.Softmax(-1))
        self.apg_eta_ob = nn.Sequential(
            nn.Linear(K+D, D))

    def forward(self, ob, z, prior_ng, sampled=True, tau_old=None, mu_old=None):
        q = probtorch.Trace()
        (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
        ob_z = torch.cat((ob, z), -1) # concatenate observations and cluster asssignemnts
        q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.apg_eta_ob(ob_z) , self.apg_eta_gamma(ob_z), prior_alpha, prior_beta, prior_mu, prior_nu)
        
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


class Enc_apg_z(nn.Module):
    """
    Conditional proposal of cluster assignments z
    """
    def __init__(self, K, D, num_hidden):
        super(self.__class__, self).__init__()
        self.pi_log_prob = nn.Sequential(
            nn.Linear(3*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

    def forward(self, ob, tau, mu, sampled=True, z_old=None):
        q = probtorch.Trace()
        gamma_list = []
        N = ob.shape[-2]
        for k in range(mu.shape[-2]):
            data_ck = torch.cat((ob, mu[:, :, k, :].unsqueeze(-2).repeat(1,1,N,1), tau[:, :, k, :].unsqueeze(-2).repeat(1, 1, N, 1)), -1) ## S * B * N * 3D
            gamma_list.append(self.pi_log_prob(data_ck))
        q_probs = F.softmax(torch.cat(gamma_list, -1), -1)
        if sampled == True:
            z = cat(q_probs).sample()
            _ = q.variable(cat, probs=q_probs, value=z, name='states')
        else:
            _ = q.variable(cat, probs=q_probs, value=z_old, name='states')
        return q
    
class Generative():
    """
    The generative model of GMM
    """
    def __init__(self, K, D, CUDA, device):
        super().__init__()
        self.K = K
        self.prior_mu = torch.zeros((K, D))
        self.prior_nu = torch.ones((K, D)) * 0.1
        self.prior_alpha = torch.ones((K, D)) * 2
        self.prior_beta = torch.ones((K, D)) * 2
        self.prior_pi = torch.ones(K) * (1./ K)

        if CUDA:
            with torch.cuda.device(device):
                self.prior_mu = self.prior_mu.cuda()
                self.prior_nu = self.prior_nu.cuda()
                self.prior_alpha = self.prior_alpha.cuda()
                self.prior_beta = self.prior_beta.cuda()
                self.prior_pi = self.prior_pi.cuda()

        self.prior_ng = (self.prior_alpha, self.prior_beta, self.prior_mu, self.prior_nu) ## this tuple is needed as parameter in enc_eta as enc_rws
    def eta_prior(self, q):
        p = probtorch.Trace()
        ## prior distributions
        p.gamma(self.prior_alpha,
                self.prior_beta,
                value=q['precisions'],
                name='precisions')
        p.normal(self.prior_mu,
                 1. / (self.prior_nu * p['precisions'].value).sqrt(),
                 value=q['means'],
                 name='means')
        return p

    def eta_sample_prior(self, S, B):
        p = probtorch.Trace()
        tau = Gamma(self.prior_alpha.unsqueeze(0).unsqueeze(0).repeat(S, B, 1, 1), self.prior_beta.unsqueeze(0).unsqueeze(0).repeat(S, B, 1, 1)).sample()
        mu = Normal(self.prior_mu.unsqueeze(0).unsqueeze(0).repeat(S, B, 1, 1), 1. / (self.prior_nu * tau).sqrt()).sample()

        p.gamma(self.prior_alpha,
                self.prior_beta,
                value=tau,
                name='precisions')
        p.normal(self.prior_mu,
                 1. / (self.prior_nu * tau).sqrt(),
                 value=mu,
                 name='means')
        return p

    def z_prior(self, q):
        p = probtorch.Trace()
        _ = p.variable(cat, probs=self.prior_pi, value=q['states'], name='states')
        return p

    def log_prob(self, ob, z , tau, mu, aggregate=False):
        """
        aggregate = False : return S * B * N
        aggregate = True : return S * B * K
        """
        sigma = 1. / tau.sqrt()
        labels = z.argmax(-1)
        labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, ob.shape[-1])
        mu_expand = torch.gather(mu, 2, labels_flat)
        sigma_expand = torch.gather(sigma, 2, labels_flat)
        ll = Normal(mu_expand, sigma_expand).log_prob(ob).sum(-1) # S * B * N
        if aggregate:
            ll = ll.sum(-1) # S * B
        return ll
    
    
