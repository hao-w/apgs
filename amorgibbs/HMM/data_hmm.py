import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import math

def sampling_hmm(T, K, D, boundary):
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    Zs_true = torch.zeros((T, K))
    mus_true = torch.zeros((K, D)).float()
    Alphas_true = torch.ones((K, K))
    for k in range(K):
        mus_true[k] = Uniform(0, boundary).sample((2,)) * torch.from_numpy(np.random.choice([-1, 1], 2)).float()
        Alphas_true[k,k] = 4.0
    covs_true = Gamma(torch.ones((K, D))*6.0, torch.ones((K, D)) * 6.0).sample()
    Pi = torch.FloatTensor([1./3, 1./3, 1./3])
    Xs = torch.zeros((T, D)).float()
    A_true = Dirichlet(Alphas_true).sample()
    for t in range(T):
        if t == 0:
            zt = cat(Pi).sample()
            label = int(zt.nonzero().item())
            xt = Normal(mus_true[label], covs_true[label]).sample()
            Xs[t] = xt
            prev_label = label
        else:
            zt = cat(A_true[prev_label]).sample()
            label = int(zt.nonzero().item())
            xt = Normal(mus_true[label], covs_true[label]).sample()
            Xs[t] = xt
            prev_label = label
        Zs_true[t] = zt
    return Xs, Zs_true, mus_true, covs_true, A_true, Pi
