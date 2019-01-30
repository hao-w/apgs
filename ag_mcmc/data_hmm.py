import numpy as np
import torch
from scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.dirichlet import Dirichlet
import math
from util import *


def sample_state(P):
    s = np.nonzero(multinomial.rvs(1, P, size=1, random_state=None)[0])[0][0]
    return s


def sampling_hmm(T, K, D):
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    Zs_true = torch.zeros((T, K))
    # A = np.array([[0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]])
    mus_true = np.array([[-5.5,-2.4], [4,6], [7, -5]])
    cov1 = np.expand_dims(np.array([[1.2, 0],[0, 1.3]]), 0)
    cov2 = np.expand_dims(np.array([[1.5, 0.3],[0.3, 1.5]]), 0)
    cov3 = np.expand_dims(np.array([[2.0, 0.8],[-0.8, 2.0]]), 0)
    covs_true = np.concatenate((cov1, cov2, cov3), axis=0) 
    Pi = np.array([1./3, 1./3, 1./3])
    Xs = torch.zeros((T, D)).float()
    mus_true = torch.from_numpy(mus_true).float()
    covs_true = torch.from_numpy(covs_true).float()
    Pi = torch.from_numpy(Pi).float()
    
    prior = initial_trans_prior(K)
    A = Dirichlet(prior).sample()
    for t in range(T):
        if t == 0:
            zt = cat(Pi).sample()
            label = torch.mm(zt.unsqueeze(0), decode_onehot).int().item()
            xt = MultivariateNormal(mus_true[label], covs_true[label]).sample()
            # xt = mus_true[label]
            Xs[t] = xt
            ztp1 = cat(A[label])
        else:
            zt = ztp1.sample()
            label = torch.mm(zt.unsqueeze(0), decode_onehot).int().item()
            xt = MultivariateNormal(mus_true[label], covs_true[label]).sample()
            # xt = mus_true[label]
            Xs[t] = xt
            ztp1 = cat(A[label])
        Zs_true[t] = zt
    return Xs, mus_true, covs_true, Zs_true, Pi, A
