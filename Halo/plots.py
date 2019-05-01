import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.gridspec as gridspec

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

def plot_rings(obs, states, K, bound):
    colors = ['r', 'b', 'g']
    assignments = states.argmax(-1)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    for k in range(K):
        obs_k = obs[np.where(assignments==k)]
        ax.scatter(obs_k[:,0], obs_k[:, 1], c=colors[k])
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])

def plot_samples(obs, q_mu, q_z, K, PATH):
    colors = ['r', 'b', 'g']

    fig = plt.figure(figsize=(25,50))
    xs = obs[0].cpu()
    batch_size, N, D = xs.shape
    E_mu = q_mu['means'].dist.loc[0].cpu().data.numpy()
    Std_mu = q_mu['means'].dist.scale[0].cpu().data.numpy()
    E_z = q_z['zs'].dist.probs[0].cpu().data.numpy()
    for b in range(batch_size):
        ax = fig.add_subplot(int(batch_size / 5), 5, b+1)
        assignments = E_z[b].argmax(-1)
        for k in range(K):
            cov_k = np.diag(Std_mu[b, k]**2)
            xk = xs[b][np.where(assignments == k)]
            ax.scatter(xk[:, 0], xk[:, 1], c=colors[k], alpha=0.2)
            plot_cov_ellipse(cov=cov_k, pos=E_mu[b, k], nstd=2, ax=ax, alpha=1.0, color=colors[k])
        ax.set_ylim([-7, 7])
        ax.set_xlim([-7, 7])
    plt.savefig('../results/modes-' + PATH + '.svg')
