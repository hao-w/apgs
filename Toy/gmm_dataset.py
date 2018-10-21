import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, multinomial
from matplotlib.patches import Ellipse
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical



def sample_state(P):
    s = np.nonzero(multinomial.rvs(1, P, size=1, random_state=None)[0])[0][0]
    return s

def sampling_hmm(T, K, D):
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
        
    Zs_true = torch.zeros((T, K))
    A = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    mus_true = np.array([[1,1], [2,10], [10, 5.5]])
    cov1 = np.expand_dims(np.array([[3, 0],[0, 0.5]]), 0)
    cov2 = np.expand_dims(np.array([[1, 0.7],[0.7, 1]]), 0)
    cov3 = np.expand_dims(np.array([[1, -0.8],[-0.8, 1]]), 0)
    covs_true = np.concatenate((cov1, cov2, cov3), axis=0) 
    Pi = np.array([1/3, 1/3, 1/3])
    
    Xs = torch.zeros((T, D)).float()
    mus_true = torch.from_numpy(mus_true).float()
    covs_true = torch.from_numpy(covs_true).float()
    Pi = torch.from_numpy(Pi).float()
    A = torch.from_numpy(A).float()
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

def sampling_gmm(T, K):
    Zs_true = np.zeros((T, K))
    mus_true = np.array([[1,1], [2,10], [10, 5.5]])
    cov1 = np.expand_dims(np.array([[3, 0],[0, 0.5]]), 0)
    cov2 = np.expand_dims(np.array([[1, 0.7],[0.7, 1]]), 0)
    cov3 = np.expand_dims(np.array([[1, -0.8],[-0.8, 1]]), 0)
    covs_true = np.concatenate((cov1, cov2, cov3), axis=0)
    Pi = np.array([1/3, 1/3, 1/3])
    Xs = []
    for t in range(T):
        zt = sample_state(Pi)
        xt = multivariate_normal.rvs(mus_true[zt], covs_true[zt])
        Xs.append(xt)
        Zs_true[t, zt] = 1
   
    Xs = np.asarray(Xs)

    return Xs, mus_true, covs_true, Zs_true

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
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    plot_cov_ellipse(cov=covs_true[0], pos=mus_true[0], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs_true[1], pos=mus_true[1], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs_true[2], pos=mus_true[2], nstd=2, ax=ax, alpha=0.5)
    plt.show()
