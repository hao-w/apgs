import torch
import torch.nn as nn
from collections.abc import Iterable
from torch.distributions.normal import Normal
import probtorch

class Enc_rws_mu_lstm(nn.Module):

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


        self.q_mu = nn.Sequential(
                nn.Linear(H, K*D))

        self.log_q_log_std = nn.Sequential(
                nn.Linear(H, K*D))

    def forward(self, ob, K, sampled=True, mu_old=None, batch_first=True):
        q = probtorch.Trace()
        S, B, N, D = ob.shape
        in_seqs = ob.reshape(S*B, N, D).transpose(0, 1)
        out_seqs, _ = self.lstm(in_seqs, self.hidden)
        out_seqs = out_seqs.transpose(0, 1).reshape(S, B, N, -1)
        out_last = out_seqs[:, :, N-1, :] #S, B, H

        q_mu_mu = self.q_mu(out_last).reshape(S, B, K, D)
        q_mu_sigma = torch.exp(self.log_q_log_std(out_last)).reshape(S, B, K, D)
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
