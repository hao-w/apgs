import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.gridspec as gridspec

def pairwise(Zs, T):
    return torch.bmm(Zs[:T-1].unsqueeze(-1), Zs[1:].unsqueeze(1))

def plot_results(EUBOs, ELBOs):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(ELBOs, 'b-', label='elbo')
    ax1.plot(EUBOs, 'r-', label='eubo')
    ax1.legend(fontsize=18)
    ax1.set_xlabel('epoch', fontsize=16)
    ax1.set_ylabel('EUBO and ELBO', fontsize=16)
    ax1.tick_params(labelsize=18)

def plot_smc_sample(Zs_true, Zs_ret):
    ret_index = torch.nonzero(Zs_ret).data.numpy()
    true_index = torch.nonzero(Zs_true).data.numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(true_index[:,0], true_index[:,1], 'ro', label='truth')
    ax.plot(ret_index[:,0], ret_index[:,1], 'bo', label='sample')
    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 0.1))
    plt.show()

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

def plot_clusters(Xs, mus_true, covs_true):
    Xs = Xs.data.numpy()
    mus_true = mus_true.data.numpy()
    covs_true = covs_true.data.numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    plot_cov_ellipse(cov=covs_true[0], pos=mus_true[0], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs_true[1], pos=mus_true[1], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs_true[2], pos=mus_true[2], nstd=2, ax=ax, alpha=0.5)
    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])
    plt.show()
