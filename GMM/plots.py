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

def plot_clusters(Xs, mus_true, covs_true, K):
    Xs = Xs.data.numpy()
    mus_true = mus_true.data.numpy()
    covs_true = covs_true.data.numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    for k in range(K):
        plot_cov_ellipse(cov=covs_true[k], pos=mus_true[k], nstd=2, ax=ax, alpha=0.5)

    ax.set_ylim([-15, 15])
    ax.set_xlim([-15, 15])
    plt.show()


def plot_results(EUBOs, ELBOs, ESSs, KLs_eta_ex, KLs_eta_in, KLs_z_ex, KLs_z_in, num_samples, gs, PATH):
    fig = plt.figure(figsize=(20, 30))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(EUBOs, 'r', label='EUBOs')
    ax1.plot(ELBOs, 'b', label='ELBOs')
    ax1.tick_params(labelsize=18)
    ax1.set_ylim([-220, -130])
    ax1.legend(fontsize=18)
    ##
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(KLs_eta_ex, '#66b3ff', label='KLs_eta_ex')
    ax2.plot(KLs_eta_in, '#ff9999', label='KLs_eta_in')
    ax2.plot(KLs_z_ex, '#99ff99', label='KLs_z_ex')
    ax2.plot(KLs_z_in, 'gold', label='KLs_z_in')
    ax2.plot(np.ones(len(KLs_z_in)) * 5, 'k', label='const=5.0')
    ax2.legend(fontsize=18)
    ax2.tick_params(labelsize=18)
    ax2.set_ylim([-1, 30])
    ##
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(np.array(ESSs) / num_samples, 'm', label='ESS')
    ax3.tick_params(labelsize=18)
    ax3.set_xlabel('epochs (%d gradient steps per epoch)' % gs, size=18)
    ax3.legend()
    plt.savefig('results/train-' + PATH + '.svg')

def plot_samples(obs, q_eta, q_z, PATH):
    colors = ['r', 'b', 'gold']
    fig = plt.figure(figsize=(25,50))
    xs = obs[0].cpu().data.numpy()
    E_pi= q_z['zs'].dist.prob[0].cpu().data.numpy()
    E_mu = q_eta['means'].dist.loc[0].cpu().data.numpy()
    E_tau = (q_eta['precisions'].dist.concentration[0] / q_eta['precisions'].dist.rate[0]).cpu().data.numpy()
    batch_size = xs.shape[0]
    for b in range(batch_size):
        ax = fig.add_subplot(int(batch_size / 5), 5, b+1)
        xb = xs[b]
        zb = zs[b]
        mu = E_mu[b].reshape(K, D)
        sigma2 = 1. / E_tau[b]
        assignments = E_z.argmax(-1)
        for k in range(K):
            cov_k = np.diag(sigma2[k])
            xk = x[np.where(assignments == k)]
            ax.scatter(xk[:, 0], xk[:, 1], c=colors[k])
            plot_cov_ellipse(cov=cov_k, pos=mu[k], nstd=2, ax=ax, alpha=0.2, color=colors[k])
        ax.set_ylim([-15, 15])
        ax.set_xlim([-15, 15])
    plt.savefig('../results/modes-' + PATH + '.svg')
