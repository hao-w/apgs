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
    input i.e. a batch of GMM of size (S, B, N, 2)
    """
    def __init__(self, K, D):
        super(self.__class__, self).__init__()

        self.rws_eta_gamma = nn.Sequential(
            nn.Linear(D, K),
            nn.Softmax(-1))

        self.rws_eta_ob = nn.Sequential(
            nn.Linear(D, D))

    def forward(self, x, prior_ng):
        q = probtorch.Trace()
        (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
        q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.rws_eta_ob(x) , self.rws_eta_gamma(x), prior_alpha, prior_beta, prior_mu, prior_nu)
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

    def forward(self, q_eta_z, x, prior_ng, extend_dir):
        (prior_alpha, prior_beta, prior_mu, prior_nu) = prior_ng
        try:
            z = q_eta_z['states'].value
        except:
            raise ValueError
        ob_z = torch.cat((x, z), -1) # concatenate observations and cluster asssignemnts
        q_alpha, q_beta, q_mu, q_nu = posterior_eta(self.apg_eta_ob(ob_z) , self.apg_eta_gamma(ob_z), prior_alpha, prior_beta, prior_mu, prior_nu)
        
        q_eta_z_new = probtorch.Trace()
        _= q_eta_z_new.variable(cat, probs=q_eta_z['states'].dist.probs, value=q_eta_z['states'].value, name='states')
            
        if extend_dir == 'backward':
            q_eta_z_new.gamma(q_alpha,
                                q_beta,
                                value=q_eta_z['precisions'].value,
                                name='precisions')
            q_eta_z_new.normal(q_mu,
                                 1. / (q_nu * q_eta_z['precisions'].value).sqrt(),
                                 value=q_eta_z['means'].value,
                                 name='means')
        elif extend_dir == 'forward':
            tau = Gamma(q_alpha, q_beta).sample()
            q_eta_z_new.gamma(q_alpha,
                                q_beta,
                                value=tau,
                                name='precisions')
            mu = Normal(q_mu, 1. / (q_nu * tau).sqrt()).sample()
            q_eta_z_new.normal(q_mu,
                                 1. / (q_nu * tau).sqrt(),
                                 value=mu,
                                 name='means')
        else:
            raise ValueError
        return q_eta_z_new


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

    def forward(self, q_eta_z, x, extend_dir):
        S, B, N, D = x.shape
        try:
            tau = q_eta_z['precisions'].value
            mu = q_eta_z['means'].value
        except:
            raise ValueError
        K = mu.shape[-2]
        mu_expand = mu.unsqueeze(2).repeat(1, 1, N, 1, 1)
        tau_expand = tau.unsqueeze(2).repeat(1, 1, N, 1, 1)
        x_expand = x.unsqueeze(-2).repeat(1, 1, 1, K, 1)
        var = torch.cat((x_expand, mu_expand, tau_expand), -1)
        assert var.shape == (S, B, N, K, 3*D)
#         gamma_list = []
#         N = x.shape[-2]
#         for k in range(mu.shape[-2]):
#             data_ck = torch.cat((x, mu[:, :, k, :].unsqueeze(-2).repeat(1,1,N,1), tau[:, :, k, :].unsqueeze(-2).repeat(1, 1, N, 1)), -1) ## S * B * N * 3D
#             gamma_list.append(self.pi_log_prob(data_ck))
        gamma_list = self.pi_log_prob(var).squeeze(-1)
        q_probs = F.softmax(gamma_list, -1)
    
        q_eta_z_new = probtorch.Trace()
        q_eta_z_new.gamma(q_eta_z['precisions'].dist.concentration,
                          q_eta_z['precisions'].dist.rate,
                          value=q_eta_z['precisions'].value,
                          name='precisions')
        q_eta_z_new.normal(q_eta_z['means'].dist.loc,
                           q_eta_z['means'].dist.scale,
                           value=q_eta_z['means'].value,
                           name='means')
        if extend_dir == 'backward':
            _ = q_eta_z_new.variable(cat, probs=q_probs, value=q_eta_z['states'].value, name='states')
        elif extend_dir == 'forward':
            z = cat(q_probs).sample()
            _ = q_eta_z_new.variable(cat, probs=q_probs, value=z, name='states')
        else:
            raise ValueError
        return q_eta_z_new
    
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
        
    def forward(self, q, x):
        """
        evaluate the log joint i.e. log p (x, z, tau, mu)
        """
        p = probtorch.Trace()
        tau = q['precisions'].value
        mu = q['means'].value
        z = q['states'].value
        p.gamma(self.prior_alpha,
                self.prior_beta,
                value=tau,
                name='precisions')
        p.normal(self.prior_mu,
                 1. / (self.prior_nu * tau).sqrt(),
                 value=mu,
                 name='means')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='states')
        
        labels_flat = z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
        mu_expand = torch.gather(mu, 2, labels_flat)
        sigma_expand = torch.gather(1. / tau.sqrt(), 2, labels_flat)
        p.normal(mu_expand, 
                 sigma_expand,
                 value=x,
                 name='lls') 
        return p
        
#     def eta_prior(self, q):
        
#         p = probtorch.Trace()
#         ## prior distributions
#         p.gamma(self.prior_alpha,
#                 self.prior_beta,
#                 value=q['precisions'],
#                 name='precisions')
        
#         p.normal(self.prior_mu,
#                  1. / (self.prior_nu * p['precisions'].value).sqrt(),
#                  value=q['means'],
#                  name='means')
#         return p

#     def ll(self, ob, z, tau, mu, aggregate=False):
#         """
#         aggregate = False : return S * B * N
#         aggregate = True : return S * B * K
#         p(x | tau, mu, z)
#         """
#         sigma = 1. / tau.sqrt()
#         labels = z.argmax(-1)
#         labels_flat = labels.unsqueeze(-1).repeat(1, 1, 1, ob.shape[-1])
#         mu_expand = torch.gather(mu, 2, labels_flat)
#         sigma_expand = torch.gather(sigma, 2, labels_flat)
#         ll = Normal(mu_expand, sigma_expand).log_prob(ob).sum(-1) # S * B * N
#         if aggregate:
#             ll = ll.sum(-1) # S * B
#         return ll
    
    
