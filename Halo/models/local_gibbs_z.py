import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Gibbs_z():
    """
    Gibbs sampling for p(z | mu, tau, x) given mu, tau, x
    """
    def __init__(self, K, CUDA, device):

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)

    def forward(self, ob, angle, mu, radi, noise_sigma, K):
        S, B, N, _ = ob.shape
        mu_expand = mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) # S * B * K * N * D
        ob_expand = ob.unsqueeze(2).repeat(1, 1, K, 1, 1) #  S * B * K * N * D
        angle_expand = angle.unsqueeze(2).repeat(1, 1, K, 1, 1) # S * B * K * N * 1
        recon_mu = torch.cat((torch.cos(angle), torch.sin(angle)), -1) * radi + mu_expand
        p_recon = Normal(recon_mu, noise_sigma).log_prob(ob_expand).permute(0, 1, 3, 4, 2)

        q_pi = F.softmax(p_recon, -1)
        q = probtorch.Trace()
        p = probtorch.Trace()
        z = cat(q_pi).sample()
        _ = q.variable(cat, probs=q_pi, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p

    def sample_prior(self, N, sample_size, batch_size):
        p_init_z = cat(self.prior_pi)
        state = p_init_z.sample((sample_size, batch_size, N,))
        return state
