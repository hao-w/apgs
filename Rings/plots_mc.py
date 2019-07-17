import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import math
from utils import shuffler, resample
from forward_backward_pcg_dec import *

class Plot_MC:
    def __init__(self, batch_size, mcmc_steps, vis_interval, fs, title_fontsize, ob_ms, mu_marker, mu_ms, opacity, bound, colors, CUDA, device):
        super().__init__()
        self.batch_size = batch_size
        self.mcmc_steps = mcmc_steps
        self.vis_interval = vis_interval
        self.num_cols = 3 + int(self.mcmc_steps / self.vis_interval)

        self.fs = fs
        self.title_fontsize = title_fontsize
        self.ob_ms = ob_ms
        self.mu_marker = mu_marker
        self.mu_ms = mu_ms
        self.opacity = opacity
        self.bound = bound
        self.colors = colors

        self.gs = gridspec.GridSpec(self.batch_size, self.num_cols)
        self.gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
        self.fig = plt.figure(figsize=(self.fs,self.fs*self.batch_size/self.num_cols))
        self.CUDA = CUDA
        self.device = device


    def sample_data(self, Data, data_ptr, sample_size):
        num_datasets = Data.shape[0]
        indices = torch.arange(num_datasets)
        batch_indices = indices[data_ptr*self.batch_size : (data_ptr+1)*self.batch_size]
        ob = Data[batch_indices]
        ob = shuffler(ob).repeat(sample_size, 1, 1, 1)
        if self.CUDA:
            with torch.cuda.device(self.device):
                ob = ob.cuda()
        return ob

    def plot_data(self, data, step, title):
        B = data.shape[0]
        for b in range(B):
            ax = self.fig.add_subplot(self.gs[b, step])
            xb = data[b]
            ax.scatter(xb[:, 0], xb[:, 1], c='k', s=self.ob_ms, alpha=self.opacity, zorder=3)
            ax.set_ylim([-self.bound, self.bound])
            ax.set_xlim([-self.bound, self.bound])
            ax.set_xticks([])
            ax.set_yticks([])
            if b == 0:
                ax.set_title(title, fontsize=self.title_fontsize)


    def plot_one_step(self, data, step, title, q_state, q_mu):
        B = data.shape[0]
        state = q_state['zs'].dist.probs[0].cpu().data.numpy()
        K = state.shape[-1]
        mu = q_mu['means'].dist.loc[0].cpu().data.numpy()
        for b in range(B):
            ax = self.fig.add_subplot(self.gs[b, step])
            xb = data[b]
            zb = state[b]
            mub = mu[b]
            assignments = zb.argmax(-1)
            for k in range(K):
                xk = xb[np.where(assignments == k)]
                ax.scatter(xk[:, 0], xk[:, 1], c=self.colors[k], s=self.ob_ms, alpha=self.opacity, zorder=3)
                ax.scatter(mub[k, 0], mub[k, 1], marker=self.mu_marker, s= self.mu_ms, c=self.colors[k])
            ax.set_ylim([-self.bound, self.bound])
            ax.set_xlim([-self.bound, self.bound])
            ax.set_xticks([])
            ax.set_yticks([])
            if b == 0:
                ax.set_title(title, fontsize=self.title_fontsize)

    def plot_results(self, models, Data, K, data_ptr):
        plt.rc('axes',edgecolor='#eeeeee')
        ob = self.sample_data(Data, data_ptr, 1)
        ob_np = ob[0].cpu().data.numpy()
        (oneshot_mu, oneshot_state, enc_angle, enc_mu, dec_x) = models
        self.plot_data(ob_np, 0, 'Data')
        state, angle, mu, w, eubo, elbo, _, q_mu, q_state, _ = oneshot(oneshot_mu, oneshot_state, enc_angle, dec_x, ob, K)
        self.plot_one_step(ob_np, 1, 'One-shot', q_state, q_mu)
        for m in range(self.mcmc_steps):
            if m == 0:
                state = resample(state, w, idw_flag=False) ## resample state
                angle = resample(angle, w, idw_flag=False)
            else:
                state = resample(state, w_state_angle, idw_flag=True)
                angle = resample(angle, w_state_angle, idw_flag=True)
            ## update mu
            mu, w_mu, eubo_mu, elbo_mu, _, q_mu  = Update_mu(enc_mu, dec_x, ob, state, angle, mu)
            mu = resample(mu, w_mu, idw_flag=True)
            ## update z
            state, angle, w_state_angle, eubo_state, elbo_state, _, q_state, q_angle, p_recon = Update_state_angle(oneshot_state, enc_angle, dec_x, ob, state, angle, mu)
            ##update angle
            if (m+1) % self.vis_interval == 0:
                self.plot_one_step(ob_np, int((m+1) / self.vis_interval)+1, 'Step %d' % (m+1), q_state, q_mu)
        recon = p_recon['likelihood'].dist.loc[0].cpu().data.numpy()
        self.plot_data(recon, -1, 'Reconstruction')

        # plt.savefig('samples.png')
