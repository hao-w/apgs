import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, multinomial
from matplotlib.patches import Ellipse
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from util import *
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import math


def sample_state(P):
    s = np.nonzero(multinomial.rvs(1, P, size=1, random_state=None)[0])[0][0]
    return s


def sampling_hmm(T, K, D):
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    Zs_true = torch.zeros((T, K))
    # A = np.array([[0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]])
    mus_true = np.array([[-3,-3], [5,20], [20, 6]])
    cov1 = np.expand_dims(np.array([[3.0, 0],[0, 2.0]]), 0)
    cov2 = np.expand_dims(np.array([[2, 0.7],[0.7, 2.5]]), 0)
    cov3 = np.expand_dims(np.array([[1.5, -0.8],[-0.8, 2.0]]), 0)
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

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def plot_samples(Xs, mus_true, covs_true):
    Xs = Xs.data.numpy()
    mus_true = mus_true.data.numpy()
    covs_true = covs_true.data.numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    # ax.set_xlim(-5, 15)
    # ax.set_ylim(-5, 15)
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    plot_cov_ellipse(cov=covs_true[0], pos=mus_true[0], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs_true[1], pos=mus_true[1], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs_true[2], pos=mus_true[2], nstd=2, ax=ax, alpha=0.5)
    plt.show()

def sampling_gmm(T, K, D):
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    Zs_true = torch.zeros((T, K))
    radius = 3.0
    angle1 = Uniform(0.0, 2 * math.pi).sample().item()
    angle2 = angle1 + 2 * math.pi / 3
    if angle2 > 2 * math.pi:
        angle2 -= 2 * math.pi
    angle3 = angle2 + 2 * math.pi / 3
    if angle3 > 2 * math.pi:
        angle3 -= 2 * math.pi
    mus_true = np.array([[math.cos(angle1) * radius, math.sin(angle1) * radius], [math.cos(angle2) * radius, math.sin(angle2) * radius], [math.cos(angle3) * radius, math.sin(angle3) * radius]])
    covs_true = np.ones((K, D)) * 0.5
    Pi = np.array([1./3, 1./3, 1./3])
    Xs = torch.zeros((T, D)).float()
    mus_true = torch.from_numpy(mus_true).float()
    covs_true = torch.from_numpy(covs_true).float()
    Pi = torch.from_numpy(Pi).float()
    
    prior = initial_trans_prior(K)
    for t in range(T):
        zt = cat(Pi).sample()
        label = torch.mm(zt.unsqueeze(0), decode_onehot).int().item()
        xt = Normal(mus_true[label], covs_true[label]).sample()
        # xt = mus_true[label]
        Xs[t] = xt
        Zs_true[t] = zt
    return Xs, mus_true, covs_true, Zs_true, Pi
