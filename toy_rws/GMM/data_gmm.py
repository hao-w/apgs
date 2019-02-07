import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
import math

def sampling_gmm(T, K, D, ind, radius, prior_covs):
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
        covs_true = Normal(torch.ones((K, D)), torch.ones((K, D)) * 0.3 ).sample()
        Xs = Normal(mus_true[labels], covs_true[labels]).sample()
    return Xs, mus_true, covs_true, Zs_true, Pi
