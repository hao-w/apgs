import torch
from torch.distributions.normal import Normal
import math
from utils import global_to_local

class Dec_x():
    def __init__(self, radi, recon_sigma, CUDA, device):
        super(self.__class__, self).__init__()
        self.recon_sigma = recon_sigma
        self.radi = radi
        if CUDA:
            with torch.cuda.device(device):
                self.recon_sigma = self.recon_sigma.cuda()
                self.radi = self.radi.cuda()

    def forward(self, ob, state, angle, mu, idw_flag=False):
        embedding = torch.cat((global_to_local(mu, state), angle), -1)
        D = mu.shape[-1]
        K = state.shape[-1]
        labels = state.argmax(-1)
        mu_expand = global_to_local(mu, state)
        recon_mu = torch.cat((torch.cos(angle), torch.sin(angle)), -1) * self.radi + mu_expand
        ll = Normal(recon_mu, self.recon_sigma).log_prob(ob).sum(-1)
        if idw_flag:
            ll = torch.cat([((labels==k).float() * ll).sum(-1).unsqueeze(-1) for k in range(K)], -1) # S * B * K
        return ll
