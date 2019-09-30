import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math

class Enc_state(nn.Module):
    def __init__(self, K, D, num_hidden, CUDA, device):
        super(self.__class__, self).__init__()
        self.pi_log_prob = nn.Sequential(
            nn.Linear(2*D, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1))


        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            with torch.cuda.device(device):
                self.prior_pi = self.prior_pi.cuda()
    def forward(self, ob, mu, K, sampled=True, state_old=None):
        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, N, D = ob.shape

        ob_mu = torch.cat((ob.unsqueeze(2).repeat(1, 1, K, 1, 1),  mu.unsqueeze(-2).repeat(1, 1, 1, N, 1)), -1) ## S * B * K * N * 2*D
        q_probs = F.softmax(self.pi_log_prob(ob_mu).squeeze(-1).transpose(-1, -2), -1)
        if sampled == True:
            state = cat(q_probs).sample()
            _ = q.variable(cat, probs=q_probs, value=state, name='states')
            _ = p.variable(cat, probs=self.prior_pi, value=state, name='states')
        else:
            _ = q.variable(cat, probs=q_probs, value=state_old, name='states')
            _ = p.variable(cat, probs=self.prior_pi, value=state_old, name='states')
        return q, p
