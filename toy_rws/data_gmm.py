import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import math

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip

def plot_samples(Xs, mus_true, covs_true):
    Xs = Xs.data.numpy()
    mus_true = mus_true.data.numpy()
    covs_true = covs_true.data.numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    plot_cov_ellipse(cov=covs_true[0], pos=mus_true[0], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs_true[1], pos=mus_true[1], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs_true[2], pos=mus_true[2], nstd=2, ax=ax, alpha=0.5)
    ax.set_ylim([-5, 5])
    ax.set_xlim([-5, 5])
    plt.show()

def sampling_gmm(T, K, D, ind):
    radius = 3.0
    mus_true = torch.zeros((K, D)).float()
    for k in range(K):
        if k == 0:
            angle = ind
        else:
            angle = angle + 2 * math.pi / K
            if angle > 2 * math.pi:
                angle -= 2 * math.pi
        mus_true[k] = torch.FloatTensor([math.cos(angle) * radius, math.sin(angle) * radius])
    covs_true = torch.eye(D).unsqueeze(0).repeat(K, 1, 1) * 0.5
    Pi = torch.FloatTensor([1./3, 1./3, 1./3])
    Zs_true = cat(Pi).sample((T,))
    labels = Zs_true.nonzero()[:, 1]
    Xs = mvn(mus_true[labels], covs_true[labels]).sample()
    return Xs, mus_true, covs_true, Zs_true, Pi
