import torch
from utils import shuffler
from apg_objective import apg_objective
from resample import Resampler
import time

def viz_dmm(ax, ob, K, mu_marker_size, marker_size, opacity, bound, colors, latents=None):
    if latents == None:
        ax.scatter(ob[:, 0], ob[:, 1], c='k', s=marker_size, zorder=3)
    else:
        (mu, z) = latents
        assignments = z.argmax(-1)
        for k in range(K):
            ob_k = ob[np.where(assignments == k)]
            ax.scatter(ob_k[:, 0], ob_k[:, 1], c=colors[k], s=marker_size, alpha=opacity, zorder=3)
            ax.scatter(mu[k, 0], mu[k, 1], marker='x', s=mu_marker_size, c=colors[k])
    ax.set_ylim([-bound, bound])
    ax.set_xlim([-bound, bound])
    ax.set_xticks([])
    ax.set_yticks([])

def viz_samples(data, trace, num_sweeps, K, viz_interval=3, figure_size=3, title_fontsize=20, mu_marker_size=150, marker_size=1.0, opacity=1.0, bound=10, colors=['#0077BB',  '#EE7733', '#AA3377', '#009988', '#BBBBBB', '#EE3377', '#DDCC77'], save_name=None):
    """
    visualize the samples along the sweeps
    """
    E_mu, E_z, E_recon = trace['E_mu'].cpu(), trace['E_z'].cpu(), trace['E_recon'].cpu()
    num_rows = len(data)
    num_cols = 3 + int(apg_sweeps-1 / viz_interval)
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(figure_size * num_cols, figure_size * num_rows))
    for row_ind in range(num_rows):
        ax = fig.add_subplot(gs[row_ind, 0])
        viz_dgmm(ax, data[row_ind], K, mu_marker_size, marker_size, opacity, bound, colors, latents=None)
        if row_ind == 0:
            ax.set_title('Data', fontsize=title_fontsize)
        for col_ind in range(num_cols-2):
            sweep = col_ind * viz_interval
            ax = fig.add_subplot(gs[row_ind, col_ind+1])
            viz_dmm(ax, data[row_ind], K, mu_marker_size, marker_size, opacity, bound, colors, latents=(E_mu[sweep, row_ind], E_z[sweep, row_ind]))
        if row_ind == 0:
            if sweep == 0:
                ax.set_title('RWS', fontsize=title_fontsize)
            else:
                ax.set_title('sweep %d' % sweep, fontsize=title_fontsize)
        ax = fig.add_subplot(gs[row_ind, num_cols-1])
        viz_dgmm(ax, recon[-1, row_ind], K, mu_marker_size, marker_size, opacity, bound, colors, latents=(E_mu[-1, row_ind], E_z[-1, row_ind]))
        if row_ind == 0:
            ax.set_title('reconstruction', fontsize=title_fontsize)
    if save_name is not None:
        plt.savefig(save_name + '.svg', dpi=300)