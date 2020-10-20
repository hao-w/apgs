import torch
import torch.nn as nn
from collections.abc import Iterable
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
import probtorch

class Enc_rws_eta_lstm(nn.Module):
    def __init__(self, K, D, H, B, S, L, CUDA, DEVICE):
        super(self.__class__, self).__init__()

        self.lstm = nn.LSTM(D, H, L)
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.hidden = (torch.zeros(L, B*S, H).cuda(),
                            torch.zeros(L, B*S, H).cuda())
        else:
            self.hidden = (torch.zeros(L, B*S, H),
                           torch.zeros(L, B*S, H))

        self.log_q_alpha = nn.Sequential(
                nn.Linear(H, K*D))

        self.log_q_beta = nn.Sequential(
                nn.Linear(H, K*D))

        self.q_mu = nn.Sequential(
                nn.Linear(H, K*D))

        self.log_q_nu = nn.Sequential(
                nn.Linear(H, K*D))
                
    def forward(self, ob, K, sampled=True, tau_old=None, mu_old=None, batch_first=True):
        q = probtorch.Trace()
        S, B, N, D = ob.shape
        in_seqs = ob.reshape(S*B, N, D).transpose(0, 1)
        out_seqs, _ = self.lstm(in_seqs, self.hidden)
        out_seqs = out_seqs.transpose(0, 1).reshape(S, B, N, -1)
        out_last = out_seqs[:, :, N-1, :] #S, B, H

        # Computing sufficient stats
        q_alpha = torch.exp(self.log_q_alpha(out_last)).reshape(S, B, K, D) # S, B, K*D
        q_beta = torch.exp(self.log_q_beta(out_last)).reshape(S, B, K, D)
        q_mu = self.q_mu(out_last).reshape(S, B, K, D)
        q_nu = torch.exp(self.log_q_nu(out_last)).reshape(S, B, K, D)
        if sampled:
            precisions = Gamma(q_alpha, q_beta).sample()
            q.gamma(q_alpha,
                    q_beta,
                    value=precisions,
                    name='precisions')
            means = Normal(q_mu, 1. / (q_nu * q['precisions'].value).sqrt()).sample()
            q.normal(q_mu,
                     1. / (q_nu * q['precisions'].value).sqrt(),
                     value=means,
                     name='means')
        else:
            q.gamma(q_alpha,
                    q_beta,
                    value=tau_old,
                    name='precisions')
            q.normal(q_mu,
                     1. / (q_nu * q['precisions'].value).sqrt(),
                     value=mu_old,
                     name='means')
        return q
