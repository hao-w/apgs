import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import math

def sampling_gmm_orbits(T, K, D, ind, radius, prior_covs):
    mus_true = torch.zeros((K, D)).float()
    for k in range(K):
        if k == 0:
            angle = ind
        else:
            angle = angle + 2 * math.pi / K
            if angle > 2 * math.pi:
                angle -= 2 * math.pi
        mus_true[k] = torch.FloatTensor([math.cos(angle) * radius, math.sin(angle) * radius])
    Pi = torch.FloatTensor([1./3, 1./3, 1./3])
    Zs_true = cat(Pi).sample((T,))
    labels = Zs_true.nonzero()[:, 1]
    if not prior_covs:
        covs_true = torch.eye(D).unsqueeze(0).repeat(K, 1, 1)
        Xs = mvn(mus_true[labels], covs_true[labels]).sample()
    else:
        covs_true = Gamma(torch.ones((K, D))*6.0, torch.ones((K, D)) * 6.0).sample()
        Xs = Normal(mus_true[labels], covs_true[labels]).sample()
    return Xs, mus_true, covs_true, Zs_true, Pi

def sampling_gmm_uniform(T, K, D, boundary):
    mus_true = torch.zeros((K, D)).float()
    for k in range(K):
        mus_true[k] = Uniform(0, boundary).sample((2,)) * torch.from_numpy(np.random.choice([-1, 1], 2)).float()
    # Pi = torch.FloatTensor([1./3, 1./3, 1./3])
    Pi = torch.FloatTensor([1./4, 1./4, 1/4., 1./4])

    Zs_true = cat(Pi).sample((T,))
    labels = Zs_true.nonzero()[:, 1]
    precisions = Gamma(torch.ones((K, D))*2.0, torch.ones((K, D)) * 2.0).sample()
    sigmas_true = 1. / torch.sqrt(precisions)
    Xs = Normal(mus_true[labels], sigmas_true[labels]).sample()
    return Xs, mus_true, sigmas_true, Zs_true, Pi

def sampling_gmm_conjugate(T, K, D):
    precisions = Gamma(torch.ones((K, D)) * 3, torch.ones((K,D)) * 3).sample()
    sigmas_true = 1. / torch.sqrt(precisions)
    mus_sigmas = sigmas_true / 0.3
    mus_true = Normal(torch.zeros((K, D)), mus_sigmas).sample()
    Pi = torch.FloatTensor([1./3, 1./3, 1./ 3])
    # Pi = torch.FloatTensor([1./4, 1./4, 1/4., 1./4])

    Zs_true = cat(Pi).sample((T,))
    labels = Zs_true.nonzero()[:, 1]
    Xs = Normal(mus_true[labels], sigmas_true[labels]).sample()
    return Xs, mus_true, sigmas_true, Zs_true, Pi
