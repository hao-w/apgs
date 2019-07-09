import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
from utils import True_decoder

class Gibbs_state():
    """
    Gibbs sampling for p(z | mu, tau, x) given mu, tau, x
    """
    def __init__(self, K, CUDA, device):

        self.prior_pi = torch.ones(K) * (1./ K)
        self.one_hots = torch.eye(K)
        if CUDA:
            with torch.cuda.device(device):
                self.prior_pi = self.prior_pi.cuda()
                self.one_hots = self.one_hots.cuda()

    def forward(self, dec_x, ob, angle, mu, K):
        S, B, N, _ = ob.shape
        lls = []
        for k in range(K):
            one_hot_k = self.one_hots[k].repeat(S, B, N, 1)
            p_k = dec_x.forward(ob, one_hot_k, angle, mu, idw_flag=False)
            lls.append(p_k.unsqueeze(-1))
            # lls.append(p_k['likelihood'].log_prob.sum(-1).unsqueeze(-1))
        q_pi = F.softmax(torch.cat(lls, -1), -1)
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
