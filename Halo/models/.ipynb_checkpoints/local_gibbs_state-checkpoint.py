import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
from utils import True_decoder

class Gibbs_state():
    """
    Gibbs sampling for p(z | mu, tau, x) given mu, tau, x
    """
    def __init__(self, K, radi, recon_sigma, CUDA, device):

        self.prior_pi = torch.ones(K) * (1./ K)
        self.radi = radi
        self.recon_sigma  = recon_sigma
        if CUDA:
            with torch.cuda.device(device):
                self.prior_pi = self.prior_pi.cuda()
                self.radi = self.radi.cuda()
                self.recon_sigma = self.recon_sigma.cuda()

    def forward(self, ob, angle, mu, K):
        S, B, N, _ = ob.shape
        ob_cetnered = ob.unsqueeze(2).repeat(1, 1, K, 1, 1) - mu.unsqueeze(-2).repeat(1, 1, 1, N, 1)
        distances = (ob_centered ** 2).sum(-1).sqrt()
        log_gammas = Normal(self.radi, self.recon_sigma).log_prob(distances).transpose(-1, -2)
        q_pi = F.softmax(log_gammas, -1)
        q = probtorch.Trace()
        p = probtorch.Trace()
        z = cat(q_pi).sample()
        _ = q.variable(cat, probs=q_pi, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p

    # def forward(self, ob, angle, mu, K):
    #     S, B, N, _ = ob.shape
    #     ob_expand = ob.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
    #     mu_expand = mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
    #     recon_mu = (torch.cat((torch.cos(angle), torch.sin(angle)), -1) * self.radi).unsqueeze(2).repeat(1, 1, K, 1, 1) + mu_expand
    #     log_gammas = Normal(recon_mu, self.recon_sigma).log_prob(ob_expand).sum(-1).transpose(-1, -2)
    #     q_pi = F.softmax(log_gammas, -1)
    #     q = probtorch.Trace()
    #     p = probtorch.Trace()
    #     z = cat(q_pi).sample()
    #     _ = q.variable(cat, probs=q_pi, value=z, name='zs')
    #     _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
    #     return q, p
