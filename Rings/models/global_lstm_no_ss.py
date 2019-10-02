import numpy as np
import torch
import torch.nn as nn
from collections.abc import Iterable
from torch.distributions.normal import Normal
import probtorch

class LSTM_eta(nn.Module):

    def __init__(self, K, D, B, S, H, L, CUDA, device):
        super(self.__class__, self).__init__()


        self.lstm = nn.LSTM(D, H, L)
        if CUDA:
            self.hidden = (torch.zeros(L, B*S, H).cuda().to(device),
                        torch.zeros(L, B*S, H).cuda().to(device))
        else:
            self.hidden = (torch.zeros(L, B*S, H),
                           torch.zeros(L, B*S, H))


        self.q_mu = nn.Sequential(
                nn.Linear(H, K*D))

        self.log_q_log_std = nn.Sequential(
                nn.Linear(H, K*D))

        self.prior_mu_mu = torch.zeros((K, D))
        self.prior_mu_sigma = torch.ones((K, D)) * 10.0

        if CUDA:
            self.prior_mu_mu = self.prior_mu_mu.cuda()
            self.prior_mu_sigma = self.prior_mu_sigma.cuda()
    def forward(self, obs, K, batch_first=True):
        q = probtorch.Trace()
        p = probtorch.Trace()
        S, B, T, D = obs.shape
        in_seqs = obs.reshape(S*B, T, D).transpose(0, 1)
        out_seqs, _ = self.lstm(in_seqs, self.hidden)
        out_seqs = out_seqs.transpose(0, 1).reshape(S, B, T, -1)
        out_last = out_seqs[:, :, T-1, :] #S, B, H

        q_mu_mu = self.q_mu(out_last).reshape(S, B, K, D)
        q_mu_sigma = torch.exp(self.log_q_log_std(out_last)).reshape(S, B, K, D)

        mu = Normal(q_mu_mu, q_mu_sigma).sample()
        q.normal(q_mu_mu,
                 q_mu_sigma,
                 value=mu,
                 name='means')

        p.normal(self.prior_mu_mu,
                 self.prior_mu_sigma,
                 value=q['means'],
                 name='means')
        return q, p
# def packSeq(Seqs, Lens, batch_size=None):
#     if batch_size is None:
#         batch_size = len(Seqs)
#     num_batches = len(Seqs) // batch_size

#     batches = []
#     for b in range(num_batches):
#         Seqs_batch_sorted = []
#         Lens_batch_sorted = []
#         # Sort Sequences
#         for l,s in sorted(zip(Lens[b*batch_size:(b+1)*batch_size],
#                                 Seqs[b*batch_size:(b+1)*batch_size]),
#                             key=lambda pair: -pair[0]):
#             Seqs_batch_sorted.append(s)
#             Lens_batch_sorted.append(l)
#         Lens_batch_sorted = torch.tensor(Lens_batch_sorted, dtype=torch.long)

#         # Pack Sequences
#         Seqs_batch_padded = torch.nn.utils.rnn.pad_sequence(Seqs_batch_sorted)
#         Seqs_batch_packed  = torch.nn.utils.rnn.pack_padded_sequence(Seqs_batch_padded, Lens_batch_sorted)
#         batches.append((Seqs_batch_packed, Lens_batch_sorted))
#     return batches
