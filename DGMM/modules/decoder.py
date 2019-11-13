import math
import torch
import torch.nn as nn
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch

class Dec(nn.Module):
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
