import torch
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

"""
==========
visualization functions
==========
"""
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

def viz_gmm(ax, ob, K, marker_size, opacity, bound, colors, latents=None):
    if latents == None:
        ax.scatter(ob[:, 0], ob[:, 1], c='k', s=marker_size, zorder=3)
    else:
        (tau, mu, state) = latents
        sigma2 = 1. / tau
        assignments = state.argmax(-1)
        for k in range(K):
            cov_k = np.diag(sigma2[k])
            ob_k = ob[np.where(assignments == k)]
            ax.scatter(ob_k[:, 0], ob_k[:, 1], c=colors[k], s=marker_size, zorder=3)
            plot_cov_ellipse(cov=cov_k, pos=mu[k], nstd=2, color=colors[k], ax=ax, alpha=opacity, zorder=3)
    ax.set_ylim([-bound, bound])
    ax.set_xlim([-bound, bound])
    ax.set_xticks([])
    ax.set_yticks([])

def viz_samples(datas, metrics, apg_sweeps, K, viz_interval, figure_size, title_fontsize, marker_size, opacity, bound, colors, save_name=None):
    """
    ==========
    visualize the samples along the sweeps
    ==========
    """
    num_rows = len(datas)
    num_cols = 2 + int(apg_sweeps / viz_interval)
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(figure_size, figure_size * num_rows / num_cols))
    for row_ind, data in enumerate(datas):
        E_tau = metrics[row_ind]['E_tau'].cpu()
        E_mu = metrics[row_ind]['E_mu'].cpu()
        E_z = metrics[row_ind]['E_z'].cpu()

        ax = fig.add_subplot(gs[row_ind, 0])
        viz_gmm(ax=ax,
                ob=data,
                K=K,
                marker_size=marker_size,
                opacity=opacity,
                bound=bound,
                colors=colors,
                latents=None) ## visualize raw dataset in the 1st column
        if row_ind == 0:
            ax.set_title('data', fontsize=title_fontsize)
        col_ind = 1
        for i in range(num_cols-1):
            sweep = i * viz_interval
            ax = fig.add_subplot(gs[row_ind, i+1])
            viz_gmm(ax=ax,
                    ob=data,
                    K=K,
                    marker_size=marker_size,
                    opacity=opacity,
                    bound=bound,
                    colors=colors,
                    latents=(E_tau[sweep, 0], E_mu[sweep, 0], E_z[sweep, 0]))
            if row_ind == 0:
                if sweep == 0:
                    ax.set_title('RWS', fontsize=title_fontsize)
                else:
                    ax.set_title('sweep %d' % sweep, fontsize=title_fontsize)
    if save_name is not None:
        plt.savefig(save_name + '.svg', dpi=300)

def viz_metrics(metrics, budget_sweeps, budget_samples, figure_size, title_fontsize, linewidth, colors):
    fig = plt.figure(figsize=(figure_size*2.5, figure_size))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(np.array(budget_sweeps), torch.Tensor(metrics['density_small']).data.numpy(), 'o-', c=colors[0], linewidth=linewidth, label=r'$\{\mu, \tau\}, \{c\}$')
    ax1.plot(np.array(budget_sweeps), torch.Tensor(metrics['density_large']).data.numpy(), 'o-', c=colors[1],  linewidth=linewidth, label=r'$\{\mu, \tau, c\}$')

    ax1.set_xticks(np.array(budget_sweeps))
    ax1.set_xticklabels(['K=%d\nL=%d' % (budget_sweeps[0], budget_samples[0]),
                         'K=%d\nL=%d' % (budget_sweeps[1], budget_samples[1]),
                         'K=%d\nL=%d' % (budget_sweeps[2], budget_samples[2]),
                         'K=%d\nL=%d' % (budget_sweeps[3], budget_samples[3]),
                         'K=%d\nL=%d' % (budget_sweeps[4], budget_samples[4])])
    ax1.tick_params(labelsize=title_fontsize)
    ax1.grid(alpha=0.4)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(np.array(budget_sweeps), torch.Tensor(metrics['ess_small']).data.numpy(), 'o-', c=colors[0],  linewidth=linewidth, label=r'$\{\mu, \tau\}, \{c\}$')
    ax2.plot(np.array(budget_sweeps), torch.Tensor(metrics['ess_large']).data.numpy(), 'o-', c=colors[1],  linewidth=linewidth, label=r'$\{\mu, \tau, c\}$')
    ax2.set_xticks(np.array(budget_sweeps))
    ax2.set_xticklabels(['K=%d\nL=%d' % (budget_sweeps[0], budget_samples[0]),
                         'K=%d\nL=%d' % (budget_sweeps[1], budget_samples[1]),
                         'K=%d\nL=%d' % (budget_sweeps[2], budget_samples[2]),
                         'K=%d\nL=%d' % (budget_sweeps[3], budget_samples[3]),
                         'K=%d\nL=%d' % (budget_sweeps[4], budget_samples[4])])
    ax2.tick_params(labelsize=title_fontsize)
    ax2.grid(alpha=0.4)
    ax2.legend(fontsize=title_fontsize)
    ax1.legend(fontsize=title_fontsize)
    ax1.set_ylabel(r'$\log \: p_\theta(x, \: z)$', fontsize=title_fontsize)
    ax2.set_ylabel('ESS / L', fontsize=title_fontsize)
