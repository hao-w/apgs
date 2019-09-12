import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from plots import plot_cov_ellipse
import torch
from forward_backward import *
import numpy as np

class Plot_MC:
    def __init__(self, models, K, D, sample_size, batch_size, mcmc_steps, vis_interval, fs, title_fontsize, ob_ms, opacity, bound, colors, CUDA, device):
        super().__init__()
        self.models = models
        self.S = sample_size
        self.K = K
        self.D = D
        self.batch_size = batch_size
        self.mcmc_steps = mcmc_steps
        self.vis_interval = vis_interval
        self.num_cols = 3 + int(self.mcmc_steps / self.vis_interval)

        self.fs = fs
        self.title_fontsize = title_fontsize
        self.ob_ms = ob_ms
        self.opacity = opacity
        self.bound = bound
        self.colors = colors

        self.gs = gridspec.GridSpec(self.batch_size, self.num_cols)
        self.gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
        self.fig = plt.figure(figsize=(self.fs,self.fs*self.batch_size/self.num_cols))
        self.CUDA = CUDA
        self.device = device

    def sample_data(self, Data, data_ptr):
        OB = []
        for i in range(self.batch_size):
            num_datasets = Data[i].shape[0]
            indices = torch.arange(num_datasets)
            ob_g = Data[i*2+1][indices[data_ptr]]
            OB.append(ob_g)
        return OB

    def plot_onestep(self, ob, row_ind, col_ind, title, latents=None):
        ax = self.fig.add_subplot(self.gs[row_ind, col_ind])
        if latents == None:
            ax.scatter(ob[:, 0], ob[:, 1], c='k', s=self.ob_ms, zorder=3)

        else:
            (mu, tau, state) = latents
            sigma2 = 1. / tau

            assignments = state.argmax(-1)
            for k in range(self.K):
                cov_k = np.diag(sigma2[k])
                ob_k = ob[np.where(assignments == k)]
                ax.scatter(ob_k[:, 0], ob_k[:, 1], c=self.colors[k], s=self.ob_ms, zorder=3)
                plot_cov_ellipse(cov=cov_k, pos=mu[k], nstd=2, color=self.colors[k], ax=ax, alpha=self.opacity, zorder=3)
        ax.set_ylim([-self.bound, self.bound])
        ax.set_xlim([-self.bound, self.bound])
        ax.set_xticks([])
        ax.set_yticks([])
        if row_ind == 0:
            ax.set_title(title, fontsize=self.title_fontsize)

    def plot_chains(self, Data, K, data_ptr):
        plt.rc('axes',edgecolor='#eeeeee')
        OB = self.sample_data(Data, data_ptr)
        for i, ob in enumerate(OB):
            self.plot_chain(ob, i)

    def plot_chain(self, ob, row_ind):
        self.plot_onestep(ob, row_ind, 0, 'Data')
        ob_input = ob.repeat(self.S, 1, 1, 1)
        if self.CUDA:
            with torch.cuda.device(self.device):
                ob_input = ob_input.cuda()
        (os_eta, enc_eta, enc_z) = self.models
        ob_tau, ob_mu, state, log_weights, q_eta, p_eta, q_z, p_z = Init_step_eta((os_eta, enc_z), ob_input, self.K, self.D, self.S, 1)
        latents = self.modes(q_eta, q_z)
        self.plot_onestep(ob, row_ind, 1, 'One-shot', latents)
        for m in range(self.mcmc_steps):
#             if m == 0:
#                 state = resample_state(state, w_f_z, idw_flag=False) ## resample state
#             else:
#                 state = resample_state(state, w_f_z, idw_flag=True)
            q_eta, p_eta, q_nu = enc_eta(ob_input, state, self.K, self.D)
            ob_tau, ob_mu, log_w_eta_f, log_w_eta_b = Incremental_eta(q_eta, p_eta, ob_input, state, self.K, self.D, ob_tau, ob_mu)
#             loss_p_q_eta, _, w_f_eta = detailed_balances(log_w_eta_f, log_w_eta_b)
#             obs_mu, obs_tau = resample_eta(obs_mu, obs_tau, w_f_eta, idw_flag=True) ## resample eta
            q_z, p_z = enc_z.forward(ob_input, ob_tau, ob_mu, self.K, self.S, 1)
            state, log_w_z_f, log_w_z_b = Incremental_z(q_z, p_z, ob_input, ob_tau, ob_mu, self.K, self.D, state)

#             loss_p_q_z, _, w_f_z = detailed_balances(log_w_z_f, log_w_z_b)
            if (m+1) % self.vis_interval == 0:
                latents = self.modes(q_eta, q_z)
                self.plot_onestep(ob, row_ind, int((m+1) / self.vis_interval)+1, 'Step %d' % (m+1), latents)

    def modes(self, q_eta, q_z):
        E_z = q_z['zs'].dist.probs[0, 0].cpu().data.numpy()
        E_mu = q_eta['means'].dist.loc[0, 0].cpu().data.numpy()
        E_tau = (q_eta['precisions'].dist.concentration[0, 0] / q_eta['precisions'].dist.rate[0, 0]).cpu().data.numpy()
        return (E_mu, E_tau, E_z)
