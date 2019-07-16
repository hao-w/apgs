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
            nn.Linear(2*D, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 1))

        self.prior_pi = torch.ones(K) * (1./ K)
        if CUDA:
            self.prior_pi = self.prior_pi.cuda().to(device)

    def forward(self, ob, mu, K):
        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, N, _ = ob.shape
        ob_mu = torch.cat((ob.unsqueeze(2).repeat(1, 1, K, 1, 1), mu.unsqueeze(-2).repeat(1, 1, 1, N, 1)), -1)
        q_probs = F.softmax(self.pi_log_prob(ob_mu).squeeze(-1).transpose(-1, -2), -1)
        # for k in range(K):
        #     data_ck = torch.cat((ob, mu[:, :, k, :].unsqueeze(-2).repeat(1,1,N,1)), -1)
        #     gamma_list.append(self.pi_log_prob(data_ck))
        # q_probs = F.softmax(torch.cat(gamma_list, -1), -1) # S * B * N * K
        z = cat(q_probs).sample()
        _ = q.variable(cat, probs=q_probs, value=z, name='zs')
        _ = p.variable(cat, probs=self.prior_pi, value=z, name='zs')
        return q, p
