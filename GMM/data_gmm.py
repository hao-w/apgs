import numpy as np
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
"""
T : number of points
K : number of clusters
D : observation dimensionality
"""
def sampling_gmm_conjugate(T, K, D):
    precisions = Gamma(torch.ones((K, D)) * 4, torch.ones((K,D)) * 4).sample()
    sigmas_true = 1. / torch.sqrt(precisions)
    mus_sigmas = sigmas_true / 0.3 ## nu
    mus_true = Normal(torch.zeros((K, D)), mus_sigmas).sample()
    Pi = torch.ones((K)) * (1. / K)
    Zs_true = cat(Pi).sample((T,))
    labels = Zs_true.nonzero()[:, 1]
    Xs = Normal(mus_true[labels], sigmas_true[labels]).sample()
    return Xs, mus_true, sigmas_true, Zs_true, Pi

def sampling_gmm_fixed_sigma(T, K, D):
    obs_mu = Normal(torch.zeros((K, D)), torch.ones((K, D)) * 3).sample()
    obs_sigma = torch.ones((K, D))
    Pi = torch.ones((K)) * (1. / K)
    states = cat(Pi).sample((T,))
    labels = states.nonzero()[:, 1]
    obs = Normal(obs_mu[labels], obs_sigma[labels]).sample()
    return obs, obs_mu, obs_sigma, states, Pi

def sampling_gmm_nonconjugate(T, K, D):
    obs_mu = Normal(torch.zeros((K, D)), torch.ones((K, D)) * 5).sample()
    obs_precision = Gamma(torch.ones((K, D)) * 3, torch.ones((K, D)) * 3).sample()
    obs_sigma = 1. / obs_precision.sqrt()
    Pi = torch.ones((K)) * (1. / K)
    states = cat(Pi).sample((T,))
    labels = states.nonzero()[:, 1]
    obs = Normal(obs_mu[labels], obs_sigma[labels]).sample()
    return obs, obs_mu, obs_sigma, states, Pi
