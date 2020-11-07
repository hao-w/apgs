import torch
import time
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import shuffle
from apgs.resampler import Resampler
from apgs.bmnist.objectives import apg_objective, bpg_objective, hmc_objective
from apgs.bmnist.hmc_sampler import HMC

def density_all_instances(models, AT, data_paths, sample_size, K, z_where_dim, z_what_dim, num_sweeps, lf_step_size, lf_num_steps, bpg_factor, CUDA, device, batch_size=10):
    densities = dict()
    shuffle(data_paths)
    data = torch.from_numpy(np.load(data_paths[0])).float()
    num_batches = int(data.shape[0] / batch_size)
    mnist_mean = torch.from_numpy(np.load('mnist_mean.npy')).float()
    mnist_mean = mnist_mean.repeat(sample_size, batch_size, K, 1, 1)
    for b in range(num_batches):
        time_start = time.time()
        x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1, 1)
        if CUDA:
            x = x.cuda().to(device)            
            mnist_mean = mnist_mean.cuda().to(device)
        S, B, T, FP, _ = x.shape
        resampler = Resampler('systematic', S, CUDA, device)
        resampler_bpg = Resampler('systematic', S*bpg_factor, CUDA, device)
        result_flags = {'loss_required' : False, 'ess_required' : False, 'mode_required' : False, 'density_required' : True}
        for lf in lf_num_steps:
            hmc_sampler = HMC(models, AT, S, B, T, K, z_where_dim, z_what_dim, num_sweeps, lf_step_size, lf_step_size, lf, CUDA, device)
            trace_hmc = hmc_objective(models, AT, x, result_flags, hmc_sampler, mnist_mean) 
            if 'HMC-RWS(L=%d, LF=%d)' % (S, lf) in densities:
                densities['HMC-RWS(L=%d, LF=%d)' % (S, lf)].append(trace_hmc['density'].mean(-1).mean(-1).cpu().numpy()[-1])
            else:
                densities['HMC-RWS(L=%d, LF=%d)' % (S, lf)] = [trace_hmc['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
        x_bpg = x.repeat(bpg_factor, 1, 1, 1, 1)
        mnist_mean_bpg = mnist_mean.repeat(bpg_factor, 1, 1, 1, 1)
        trace_bpg = bpg_objective(models, AT, x_bpg, result_flags, num_sweeps, resampler_bpg, mnist_mean_bpg)
        if 'BPG(L=%d)' % (S*bpg_factor) in densities:
            densities['BPG(L=%d)' % (S*bpg_factor)].append(trace_bpg['density'].mean(-1).mean(-1).cpu().numpy()[-1])
        else:
            densities['BPG(L=%d)' % (S*bpg_factor)] = [trace_bpg['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
        trace_apg = apg_objective(models, AT, x, K, result_flags, num_sweeps, resampler, mnist_mean)
        if 'APG(L=%d)' % S in densities:
            densities['APG(L=%d)' % S].append(trace_apg['density'].mean(-1).mean(-1).cpu().numpy()[-1])
        else:
            densities['APG(L=%d)' % S] = [trace_apg['density'].mean(-1).mean(-1).cpu().numpy()[-1]]
        time_end = time.time()
        print('%d / %d completed in (%ds)' % (b+1, num_batches, time_end-time_start))
    for key in densities.keys():
        densities[key] = np.array(densities[key]).mean()
        print('method=%s, log joint=%.2f' % (key, densities[key]))


def viz_samples(frames, metrics, num_sweeps, K, fs=2, title_fontsize=12, lw=2, colors=['#AA3377', '#EE7733', '#009988', '#0077BB', '#BBBBBB', '#EE3377', '#DDCC77']):
    B, T, FP, _ = frames.shape
    recons = metrics['E_recon'][-1].squeeze(0).cpu() # B * T * 96 *96
    z_wheres = metrics['E_where'][-1].squeeze(0).cpu().clone()
    z_wheres[:,:,:,1] =  z_wheres[:,:,:,1] * (-1)
    c_pixels = z_wheres
    c_pixels = (c_pixels + 1.0) * (96 - 28) / 2. # B * T * K * D
    for b in range(B):
        num_cols = T
        num_rows =  2
        c_pixel, recon = c_pixels[b].numpy(), recons[b].numpy()
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.05 , bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.2)
        fig = plt.figure(figsize=(fs * num_cols, fs * num_rows))
        for c in range(num_cols):
            ax_infer = fig.add_subplot(gs[0, c])
            ax_infer.imshow(frames[b, c].numpy(), cmap='gray', vmin=0.0, vmax=1.0)
            ax_infer.set_xticks([])
            ax_infer.set_yticks([])
            for k in range(K):
                rect_k = patches.Rectangle((c_pixel[c, k, :]), 27, 27, linewidth=lw, edgecolor=colors[k],facecolor='none')
                ax_infer.add_patch(rect_k)
            ax_recon = fig.add_subplot(gs[1, c])
            ax_recon.imshow(recon[c], cmap='gray', vmin=0.0, vmax=1.0)
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])
            if c == 0:
                ax_infer.set_title('Inferred Positions')
                ax_recon.set_title('Reconstruction')
 