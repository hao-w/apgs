import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch

class Enc_rws_mu(nn.Module):
    def __init__(self, K, D, num_hidden, num_nss):
        super(self.__class__, self).__init__()
        self.nss1 = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_nss))

        self.nss2 = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K),
            nn.Softmax(-1))

        self.mean_mu = nn.Sequential(
            nn.Linear(num_nss+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_nss+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))

    def forward(self, ob, K, priors, sampled=True, mu_old=None, EPS=1e-8):
        q = probtorch.Trace()
        S, B, N, D = ob.shape
        (prior_mu, prior_sigma) = priors
        nss1 = self.nss1(ob).unsqueeze(-1).repeat(1, 1, 1, 1, K).transpose(-1, -2)
        nss2 = self.nss2(ob).unsqueeze(-1).repeat(1, 1, 1, 1, nss1.shape[-1])
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
    
class Enc_apg_local(nn.Module):
    def __init__(self, K, D, num_hidden):
        super(self.__class__, self).__init__()

        self.pi_log_prob = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1))

        self.angle_log_con1 = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.angle_log_con0 = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

    def forward(self, ob, mu, K, sampled=True, z_old=None, beta_old=None):
        q = probtorch.Trace()
        S, B, N, D = ob.shape
        ob_mu = ob.unsqueeze(2).repeat(1, 1, K, 1, 1) - mu.unsqueeze(-2).repeat(1, 1, 1, N, 1)
        q_probs = F.softmax(self.pi_log_prob(ob_mu).squeeze(-1).transpose(-1, -2), -1)
        if sampled:
            z = cat(q_probs).sample()
            _ = q.variable(cat,
                           probs=q_probs,
                           value=z,
                           name='states')
            mu_expand = torch.gather(mu, -2, z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, D))
            q_angle_con1 = self.angle_log_con1(ob - mu_expand).exp()
            q_angle_con0 = self.angle_log_con0(ob - mu_expand).exp()
            beta = Beta(q_angle_con1, q_angle_con0).sample()
            q.beta(q_angle_con1,
                   q_angle_con0,
                   value=beta,
                   name='angles')
        else:
            _ = q.variable(cat,
                           probs=q_probs,
                           value=z_old,
                           name='states')
            mu_expand = torch.gather(mu, -2, z_old.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, D))
            q_angle_con1 = self.angle_log_con1(ob - mu_expand).exp()
            q_angle_con0 = self.angle_log_con0(ob - mu_expand).exp()
            q.beta(q_angle_con1,
                   q_angle_con0,
                   value=beta_old,
                   name='angles')
        return q

class Enc_apg_mu(nn.Module):
    def __init__(self, K, D, num_hidden, num_nss):
        super(self.__class__, self).__init__()
        self.nss1 = nn.Sequential(
            nn.Linear(D+K, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_nss))
        self.nss2 = nn.Sequential(
            nn.Linear(D+K, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, K),
            nn.Softmax(-1))
        self.mean_mu = nn.Sequential(
            nn.Linear(num_nss+2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))
        self.mean_log_sigma = nn.Sequential(
            nn.Linear(num_nss+2*D, num_hidden),
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
    
    
    
class Decoder(nn.Module):
    def __init__(self, K, D, num_hidden, recon_sigma, CUDA, DEVICE):
        super(self.__class__, self).__init__()
        self.recon_mu = nn.Sequential(
            nn.Linear(1, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, D))
        ## prior of center variable mu
        self.prior_mu_mu = torch.zeros(D)
        self.prior_mu_sigma = torch.ones(D) * 10.0
        ## prior of cluster assignemnt variable z
        self.prior_pi = torch.ones(K) * (1./ K)
        ## priors of angle variables beta
        self.prior_con1 = torch.ones(1)
        self.prior_con0 = torch.ones(1)
        ## parameters of decoders
        self.recon_sigma = torch.ones(1) * recon_sigma
        self.radi = torch.ones(1)
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_mu_mu = self.prior_mu_mu.cuda()
                self.prior_mu_sigma = self.prior_mu_sigma.cuda()
                self.prior_pi = self.prior_pi.cuda()
                self.prior_con1 = self.prior_con1.cuda()
                self.prior_con0 = self.prior_con0.cuda()
                self.recon_sigma = self.recon_sigma.cuda()
                self.radi = self.radi.cuda()


        self.radi = nn.Parameter(self.radi)

    def forward(self, ob, mu, z, beta):
        p = probtorch.Trace()
        S, B, N, D = ob.shape

        p.normal(self.prior_mu_mu,
                 self.prior_mu_sigma,
                 value=mu,
                 name='means')
        _ = p.variable(cat,
                       probs=self.prior_pi,
                       value=z,
                       name='states')
        p.beta(self.prior_con1,
               self.prior_con0,
               value=beta,
               name='angles')
        hidden = self.recon_mu(beta * 2 * math.pi)
        hidden2 = hidden / (hidden**2).sum(-1).unsqueeze(-1).sqrt()
        mu_expand = torch.gather(mu, -2, z.argmax(-1).unsqueeze(-1).repeat(1, 1, 1, D))
        recon_mu = hidden2 * self.radi + mu_expand
        p.normal(recon_mu,
                 self.recon_sigma.repeat(S, B, N, D),
                 value=ob,
                 name='likelihood')
        return p
