import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
"""
==========
visualization functions
==========
"""
def viz_dgmm(ax, ob, K, mu_marker_size, marker_size, opacity, bound, colors, latents=None):
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




def viz_samples(datas, metrics, apg_sweeps, K, viz_interval, figure_size, title_fontsize, mu_marker_size, marker_size, opacity, bound, colors, save_name=None):
    """
    ==========
    visualize the samples along the sweeps
    ==========
    """
    num_rows = len(datas)
    num_cols = 3 + int(apg_sweeps / viz_interval)
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(figure_size, figure_size * num_rows / num_cols))
    for row_ind, data in enumerate(datas):
        E_mu = metrics[row_ind]['E_mu'].cpu()
        E_z = metrics[row_ind]['E_z'].cpu()
        recon = metrics[row_ind]['E_recon'].cpu()
        ax = fig.add_subplot(gs[row_ind, 0])

        viz_dgmm(ax=ax,
                 ob=data,
                 K=K,
                 mu_marker_size=mu_marker_size,
                 marker_size=marker_size,
                 opacity=opacity,
                 bound=bound,
                 colors=colors,
                 latents=None)
        if row_ind == 0:
            ax.set_title('data', fontsize=title_fontsize)
        for i in range(num_cols-1):
            ax = fig.add_subplot(gs[row_ind, i+1])
            if i == num_cols - 2:
                viz_dgmm(ax=ax,
                         ob=recon[-1, 0],
                         K=K,
                         mu_marker_size=mu_marker_size,
                         marker_size=marker_size,
                         opacity=opacity,
                         bound=bound,
                         colors=colors,
                         latents=(E_mu[-1, 0], E_z[-1, 0]))
            else:
                sweep = i * viz_interval
                viz_dgmm(ax=ax,
                         ob=data,
                         K=K,
                         mu_marker_size=mu_marker_size,
                         marker_size=marker_size,
                         opacity=opacity,
                         bound=bound,
                         colors=colors,
                         latents=(E_mu[sweep, 0], E_z[sweep, 0]))
            if row_ind == 0:
                if i == 0:
                    ax.set_title('rws-mlp', fontsize=title_fontsize)
                elif i == num_cols-2:
                    ax.set_title('reconstruction', fontsize=title_fontsize)
                else:
                    ax.set_title('sweep %d' % sweep, fontsize=title_fontsize)


    if save_name is not None:
        plt.savefig(save_name +'.svg', dpi=300)


def Plot_metrics(ll, ess, sample_size, filename):
    num_cols = len(ll)
    gs = gridspec.GridSpec(2, num_cols)
    gs.update(left=0.05 , bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.05)
    fig = plt.figure(figsize=(self.fs, self.fs * 2 / num_cols))
    plt.rc('axes',edgecolor='#eeeeee')
    for col_ind in range(num_cols):
        ax1 = fig.add_subplot(gs[0, col_ind])
        ax2 = fig.add_subplot(gs[1, col_ind])
        temp = ll[col_ind][0].data.numpy()

        baseline = np.ones(temp.shape[0]) * temp[0]
        ax1.plot(ll[col_ind][0].data.numpy(), c=self.colors[0], marker='o')
        ax1.plot(baseline, '--', c=self.colors[1], alpha=0.4)

        ax2.plot(ess[col_ind][0].data.numpy() / sample_size, c=self.colors[2])
        ax2.scatter(np.arange(temp.shape[0]), ess[col_ind][0].data.numpy() / sample_size, c=self.colors[2], s=6.0)

        # ax1.set_title('N= %d' % ((col_ind+1) * 20), fontsize=self.title_fontsize)
        # if col_ind == 0:
            # ax1.set_ylabel('log p(z, x)', fontsize=self.title_fontsize)
            # ax2.set_ylabel('ESS / L', fontsize=self.title_fontsize)
        ax2.set_ylim([-0.1, 1.1])
        ax1.set_xticks([])
        ax1.set_ylim([temp.min()-50, temp.max()+10])
    plt.savefig(filename + '.svg', dpi=300)
    plt.savefig(filename + '.pdf')
