import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.gridspec as gridspec
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
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip

def plot_rings(obs, states, K, bound):
    colors = ['r', 'b', 'g', 'k', 'y']
    assignments = states.argmax(-1)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    for k in range(K):
        obs_k = obs[np.where(assignments==k)]
        ax.scatter(obs_k[:,0], obs_k[:, 1], s=5, alpha=0.8)
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])

def plot_samples(obs, q_mu, q_z, K, PATH):
    colors = ['r', 'b', 'g', 'm', 'y']

    fig = plt.figure(figsize=(25,25))
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
            ax.scatter(xk[:, 0], xk[:, 1], c=colors[k], s=150, alpha=0.8)
            plot_cov_ellipse(cov=cov_k, pos=E_mu[b, k], nstd=2, ax=ax, alpha=1.0, color=colors[k])
        ax.set_ylim([-7, 7])
        ax.set_xlim([-7, 7])
    plt.savefig('../results/modes-' + PATH + '.svg')

def plot_recon(recon, path, page_width, bound):
    S, B, N, D = recon.shape
    gs = gridspec.GridSpec(int(B / 5), 5)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(page_width,page_width*4/5))
    for b in range(B):
        ax = fig.add_subplot(gs[int(b / 5), int(b % 5)])
        ax.scatter(recon[0, b, :, 0], recon[0, b, :, 1], s=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(-bound, bound)
        ax.set_xlim(-bound, bound)
    plt.savefig('../results/reconstruction-' + path + '.png')
    
def plot_angles(angle_infer, angle_true, page_width):
    S, B, N, D = angle_true.shape
    angle_infer = angle_infer.cpu().data.numpy()
    angle_true = angle_true.cpu().data.numpy()
    gs = gridspec.GridSpec(int(B / 5), 5)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.2, hspace=0.2)
    fig = plt.figure(figsize=(page_width,page_width*4/5))
    for b in range(B):
        ax = fig.add_subplot(gs[int(b / 5), int(b % 5)])
        ax.scatter(angle_true[0, b, :, 0], angle_infer[0, b, :, 0], s=5)
        ax.set_ylim(0, 2*math.pi)
        ax.set_xlim(0, 2*math.pi)