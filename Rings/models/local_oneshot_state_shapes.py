import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Oneshot_state(nn.Module):
    def __init__(self, K, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.pi_log_prob = nn.Sequential(
            nn.Linear(D+1, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1))
        
        self.one_hot = torch.arange(K).float()
        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            with torch.cuda.device(device):
                self.prior_pi = self.prior_pi.cuda()
                self.one_hot = self.one_hot.cuda()
    def forward(self, ob, mu, K, sampled=True, state_old=None):
        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, N, D = ob.shape
        ob_mu = ob.unsqueeze(2).repeat(1, 1, K, 1, 1) - mu.unsqueeze(-2).repeat(1, 1, 1, N, 1) ## S * B * K * N * D
        # ob_mu = torch.atan2(ob_mu[:,:,:,:,0], ob_mu[:,:,:,:,1]).unsqueeze(-1)
        one_hots = self.one_hot.repeat(S, B, N, 1).transpose(-1, -2).unsqueeze(-1)
        
        q_probs = F.softmax(self.pi_log_prob(torch.cat((ob_mu, one_hots), -1)).squeeze(-1).transpose(-1, -2), -1)
        if sampled == True:
            state = cat(q_probs).sample()
            _ = q.variable(cat, probs=q_probs, value=state, name='states')
            _ = p.variable(cat, probs=self.prior_pi, value=state, name='states')
        else:
            _ = q.variable(cat, probs=q_probs, value=state_old, name='states')
            _ = p.variable(cat, probs=self.prior_pi, value=state_old, name='states')
        return q, p
