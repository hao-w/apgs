import os
import time
import torch
import numpy as np
from resample import Resampler
from apgs.dmm.objectives import apg_objective, bpg_objective, hmc_objective
from apgs.dmm.hmc_sampler import HMC

    
def density_all_instances(models, data, sample_size, K, num_sweeps, lf_step_size, lf_num_steps, bpg_factor, CUDA, device, batch_size=100):
    densities = dict()
    num_batches = (data.shape[0] / batch_size)
    for b in range(num_batches):
        x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1)
        if CUDA:
            x = x.cuda().to(device)            
        S, B, N, D = x.shape
        resampler = Resampler('systematic', S, CUDA, device)
        resampler_bpg = Resampler('systematic', S*bpg_factor, CUDA, device)
        result_flags = {'loss_required' : False, 'ess_required' : False, 'mode_required' : False, 'density_required' : True}
        for lf in lf_num_steps:
            hmc_sampler = HMC(S, B, N, K, D, num_sweeps, lf_step_size, lf, CUDA, device)
            trace_hmc = hmc_objective(models, x, result_flags, hmc_sampler) 
            if 'HMC-RWS(L=%d, LF=%d)' % (S, lf) in densities:
                densities['HMC-RWS(L=%d, LF=%d)' % (S, lf)].append(trace_hmc['density'].mean(-1).mean(-1).cpu().numpy()[-1])
            else:
                densities['HMC-RWS(L=%d, LF=%d)' % (S, lf)] = [trace_hmc['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
        x_bpg = x.repeat(bpg_factor, 1, 1, 1)
        trace_bpg = bpg_objective(models, x_bpg, result_flags, num_sweeps, resampler_bpg)
        if 'BPG(L=%d)' % (S*bpg_factor) in densities:
            densities['BPG(L=%d)' % (S*bpg_factor)].append(trace_bpg['density'].mean(-1).mean(-1).cpu().numpy()[-1])
        else:
            densities['BPG(L=%d)' % (S*bpg_factor)] = [trace_bpg['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
        trace_apg = apg_objective(models, x, K, result_flags, num_sweeps, resampler)
        if 'APG(L=%d)' % S in densities:
            densities['APG(L=%d)' % S].append(trace_apg['density'].mean(-1).mean(-1).cpu().numpy()[-1])
        else:
            densities['APG(L=%d)' % S] = [trace_apg['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
    for key in densities.keys():
        densities[key] = np.concatenate(densities[key], 0).mean()
        print('method=%s, log joint=%.2f' % (key, densities[key]))
    return densities



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