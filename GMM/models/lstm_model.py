import numpy as np
import torch
import torch.nn as nn
from collections.abc import Iterable
from normal_gamma import *
import probtorch

def initialize(K, D, B, S, HG, HL, L, CUDA, device, LR):
    enc_eta = LSTM_eta(K, D, B, S, HG, L, CUDA, device)
    enc_z = Enc_z(K, D, HL, CUDA, device)
    if CUDA:
        enc_eta.cuda().to(device)
        enc_z.cuda().to(device)
    optimizer =  torch.optim.Adam(list(enc_z.parameters())+list(enc_eta.parameters()),lr=LR, betas=(0.9, 0.99))
    return enc_eta, enc_z, optimizer

class LSTM_eta(nn.Module):

    def __init__(self, K, D, B, S, H, L, CUDA, device):
        super(self.__class__, self).__init__()
        self.lstm = nn.LSTM(D, H, L)
        self.hidden = (torch.zeros(L, B*S, H).cuda().to(device),
                       torch.zeros(L, B*S, H).cuda().to(device))

        self.gamma = nn.Sequential(
            nn.Linear(H, K),
            nn.Softmax(-1))

        self.ob = nn.Sequential(
            nn.Linear(H, D))

        self.prior_mu = torch.zeros((K, D))
        self.prior_nu = torch.ones((K, D)) * 0.3
        self.prior_alpha = torch.ones((K, D)) * 4
        self.prior_beta = torch.ones((K, D)) * 4
        if CUDA:
            self.prior_mu = self.prior_mu.cuda().to(device)
            self.prior_nu = self.prior_nu.cuda().to(device)
            self.prior_alpha = self.prior_alpha.cuda().to(device)
            self.prior_beta = self.prior_beta.cuda().to(device)

    def forward(self, obs, K, D, batch_first=True):
        S, B, T, D = obs.shape
        in_seqs = obs.reshape(S*B, T, D).transpose(0, 1)
        out_seqs, self.hidden = self.lstm(in_seqs, self.hidden)
        out_seqs = out_seqs.transpose(0, 1).reshape(S, B, T, -1)
        # out_last = out_seqs[:, :, T-1, :]
        # Computing sufficient stats
        gammas = self.gamma(out_seqs)
        xs = self.ob(out_seqs)
        #Computing true post params
        q_alpha, q_beta, q_mu, q_nu = Post_eta(xs, gammas, self.prior_alpha, self.prior_beta, self.prior_mu, self.prior_nu, K, D)
        precisions = Gamma(q_alpha, q_beta).sample()
        #Constructing proposal and target traces
        q = probtorch.Trace()
        p = probtorch.Trace()
        q.gamma(q_alpha,
                q_beta,
                value=precisions,
                name='precisions')
        p.gamma(self.prior_alpha,
                self.prior_beta,
                value=q['precisions'],
                name='precisions')
        means = Normal(q_mu, 1. / (q_nu * q['precisions'].value).sqrt()).sample()
        q.normal(q_mu,
                 1. / (q_nu * q['precisions'].value).sqrt(),
                 value=means,
                 name='means')
        p.normal(self.prior_mu,
                 1. / (self.prior_nu * p['precisions'].value).sqrt(),
                 value=q['means'],
                 name='means')
        return q, p, q_nu

    def sample_prior(self, sample_size, batch_size):
        p_tau = Gamma(self.prior_alpha, self.prior_beta)
        obs_tau = p_tau.sample((sample_size, batch_size,))
        p_mu = Normal(self.prior_mu.repeat(sample_size, batch_size, 1, 1), 1. / (self.prior_nu * obs_tau).sqrt())
        obs_mu = p_mu.sample()
        obs_sigma = 1. / obs_tau.sqrt()
        return obs_mu, obs_sigma

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
