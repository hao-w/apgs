import torch
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
"""
==========
visualization functions
==========
"""
def plot_baselines(flags, fs, fs_title, opacity, lw, colors):
    fig = plt.figure(figsize=(fs*2.5,fs))
    ax = fig.add_subplot(111)
    if flags['apg']:
        APG = np.load('log_joint_apg.npy')
        APG_mean = APG.mean(0)
        APG_std = APG.std(0)
        ax.plot(APG_mean, linewidth=lw, c=colors[0], label='APG(L=100)')
        ax.fill_between(np.arange(APG_mean.shape[0]), APG_mean-APG_std, APG_mean+APG_std, color=colors[0], alpha=opacity)
    if flags['hmc']:
        HMC = np.load('log_joint_hmc.npy')
        HMC_mean = HMC.mean(0)
        HMC_std = HMC.std(0)
        ax.plot(HMC_mean, linewidth=lw, c=colors[1], label='HMC-RWS(L=100, LF=5)')
        ax.fill_between(np.arange(HMC_mean.shape[0]), HMC_mean - HMC_std, HMC_mean + HMC_std, color=colors[1], alpha=opacity)
    if flags['bpg']:
        BPG = np.load('log_joint_bpg.npy')
        BPG_mean = BPG.mean(0)
        BPG_std = BPG.std(0)
        ax.plot(BPG_mean, linewidth=lw, c=colors[2], label='BPG(L=100)')
        ax.fill_between(np.arange(BPG_mean.shape[0]), BPG_mean-BPG_std, BPG_mean+BPG_std, color=colors[2], alpha=opacity)
    ax.legend(fontsize=20, loc='lower right')
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Sweeps', fontsize=25)
    ax.set_ylabel(r'$\log \: p_\theta(x, z)$', fontsize=25)
    ax.grid(alpha=0.4)


def viz_samples(frames, metrics, apg_sweeps, K, viz_interval, figure_size, title_fontsize, linewidth, colors, save_name=None):
    recon = metrics['E_recon'][-1].squeeze(0).cpu() # T * 96 *96
    z_where = metrics['E_where'][-1].squeeze(0).cpu().clone()
    z_where[:,:,1] =  z_where[:,:,1] * (-1)
    c_pixels = z_where
    c_pixels = (c_pixels + 1.0) * (96 - 28) / 2. # T * K * D
    T = metrics['E_recon'][-1].squeeze(0).shape[0]
    num_cols = T
    num_rows =  2
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.05 , bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
    fig = plt.figure(figsize=(figure_size, figure_size * num_rows / num_cols))
    for c in range(num_cols):
        ax_infer = fig.add_subplot(gs[0, c])
        ax_infer.imshow(frames[c], cmap='gray')
        ax_infer.set_xticks([])
        ax_infer.set_yticks([])
        for k in range(K):
            rect_k = patches.Rectangle((c_pixels[c, k, :]), 27, 27, linewidth=linewidth, edgecolor=colors[k],facecolor='none')
            ax_infer.add_patch(rect_k)
        ax_recon = fig.add_subplot(gs[1, c])
        ax_recon.imshow(recon[c], cmap='gray', vmax=1.0)
        ax_recon.set_xticks([])
        ax_recon.set_yticks([])
    if save_name is not None:
        plt.savefig(save_name + '.svg', dpi=300)
# class Viz_MC:
#     """
#     A class which visualizes the prediction as a Markov chain
#         Plot_long_video : used to visualize a single video with large timescales (e.g. T=100)
#         Plot_chains : align all visualizaztion of all datasets in different rows
#     """
#     def __init__(self, K, viz_interval, FS, TITLE_FS, COLORS, LW):
#         super().__init__()
#         self.K = K
#         self.viz_interval = viz_interval
#         self.FS = FS
#         self.TITLE_FS = TITLE_FS
#         self.COLORS = COLORS
#         self.LW = LW

#     def Plot_long_video(self, Metrics, PATH, save_flag=False):
#         Data = Metrics['data']
#         recons = Metrics['recon'][-1].mean(0).squeeze(0) # T * 96 *96
#         z_where = Metrics['E_where'][-1].squeeze(0).clone()
#         z_where[:,:,1] =  z_where[:,:,1] * (-1)
#         c_pixels = z_where
#         c_pixels = (c_pixels + 1.0) * (96 - 28) / 2. # T * K * D

#         num_cols = 20
#         num_rows =  2*5
#         gs = gridspec.GridSpec(num_rows, num_cols)
#         gs.update(left=0.05 , bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
#         fig = plt.figure(figsize=(self.FS, self.FS * num_rows / num_cols))
#         for r in range(5):
#             for c in range(num_cols):
#                 ax = fig.add_subplot(gs[(r*2), c])
#                 ax.imshow(Data[r*20+c], cmap='gray')
#     #             if c == 0:
#     #                 ax.set_ylabel('Data', fontsize=TITLE_FS)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 for k in range(self.K):
#                     rect_k = patches.Rectangle((c_pixels[r*20+c, k, :]), 27, 27, linewidth=self.LW, edgecolor=self.COLORS[k],facecolor='none')
#                     ax.add_patch(rect_k)

#                 ax = fig.add_subplot(gs[r*2+1, c])
#                 ax.imshow(recons[r*20+c], cmap='gray')
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#         if save_flag:
#             plt.savefig(PATH + 'figure.svg', dpi=300)

#     def Save_imgs(self, Metrics, PATH, FS):
#         Data = Metrics['data']
#         recons = Metrics['recon'][-1].mean(0).squeeze(0) # T * 96 *96
#         z_where = Metrics['E_where'][-1].squeeze(0).clone()
#         z_where[:,:,1] =  z_where[:,:,1] * (-1)
#         c_pixels = z_where
#         c_pixels = (c_pixels + 1.0) * (96 - 28) / 2. # T * K * D


#         if not os.path.exists(PATH):
#             os.mkdir(PATH)
# #             os.mkdir(PATH + 'data/')
#             os.mkdir(PATH + 'recon/')
#             os.mkdir(PATH + 'tracking/')
#         for t in range(Data.shape[0]):
# #             fig = plt.figure(figsize=(FS, FS))
# #             ax = fig.add_subplot(111)
# #             ax.set_xticks([])
# #             ax.set_yticks([])
# #             ax.imshow(Data[t], cmap='gray')
# #             plt.savefig(PATH + 'data/%03d.png' % (t+1), dpi=300)
# #             plt.close()
#             fig = plt.figure(figsize=(FS,FS)) # recon
#             ax = fig.add_subplot(111)
#             ax.set_axis_off()

#             ax.set_xticks([])
#             ax.set_yticks([])
# #             for k in range(self.K):
# #                 rect_k = patches.Rectangle((c_pixels[t, k, :]), 27, 27, linewidth=self.LW, edgecolor=self.COLORS[k],facecolor='none')
# #                 ax.add_patch(rect_k)
#             ax.imshow(recons[t], cmap='gray')
#             plt.savefig(PATH + 'recon/%03d.png' % (t+1), dpi=300,bbox_inches='tight', pad_inches=0)
#             plt.close()

#             fig = plt.figure(figsize=(FS,FS)) # tracking
#             ax = fig.add_subplot(111)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_axis_off()

#             for k in range(self.K):
#                 rect_k = patches.Rectangle((c_pixels[t, k, :]), 27, 27, linewidth=self.LW, edgecolor=self.COLORS[k],facecolor='none')
#                 ax.add_patch(rect_k)
#             ax.imshow(Data[t], cmap='gray')
#             plt.savefig(PATH + 'tracking/%03d.png' % (t+1), dpi=300,bbox_inches='tight', pad_inches=0)
#             plt.close()
