import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import numpy as np

class Viz_MC:
    """
    A class which visualizes the prediction as a Markov chain
        Plot_onestep : visualize the data/sample in one time Step
        Plot_chains : align all visualizaztion of all datasets in different rows
    """
    def __init__(self, K, viz_interval, fs, title_fontsize, ob_ms, mu_marker,mu_ms, opacity, bound, colors):
        super().__init__()
        self.K = K
        self.viz_interval = viz_interval
        self.fs = fs
        self.title_fontsize = title_fontsize
        self.ob_ms = ob_ms
        self.mu_marker = mu_marker
        self.mu_ms = mu_ms
        self.opacity = opacity
        self.bound = bound
        self.colors = colors

    def Plot_onestep(self, ax, ob, latents=None):
        if latents == None:
            ax.scatter(ob[:, 0], ob[:, 1], c='k', s=self.ob_ms, zorder=3)
        else:
            (mu, state) = latents
            assignments = state.argmax(-1)
            for k in range(self.K):
                ob_k = ob[np.where(assignments == k)]
                ax.scatter(ob_k[:, 0], ob_k[:, 1], c=self.colors[k], s=self.ob_ms, alpha=self.opacity, zorder=3)
                ax.scatter(mu[k, 0], mu[k, 1], marker=self.mu_marker, s= self.mu_ms, c=self.colors[k])
        ax.set_ylim([-self.bound, self.bound])
        ax.set_xlim([-self.bound, self.bound])
        ax.set_xticks([])
        ax.set_yticks([])

    def Plot_chains(self, data_list, sample_lists, recon_lists):
        ## initialize figure object
        num_rows = len(data_list)
        num_steps = len(sample_lists[0])
        num_cols = 3 + int((num_steps - 1)/ self.viz_interval)
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
        fig = plt.figure(figsize=(self.fs, self.fs * num_rows / num_cols))
        plt.rc('axes',edgecolor='#eeeeee')
        for row_ind, sample_list in enumerate(sample_lists):
            data = data_list[row_ind].data.numpy()
            recon_list = recon_lists[row_ind]
            ax = fig.add_subplot(gs[row_ind, 0])
            self.Plot_onestep(ax, data) ## visualize raw dataset in the 1st column

            if row_ind == 0:
                ax.set_title('Data', fontsize=self.title_fontsize)
            col_ind = 1
            for i in range(0, num_steps, self.viz_interval):
                ax = fig.add_subplot(gs[row_ind, col_ind])
                self.Plot_onestep(ax, recon_list[i], latents=sample_list[i]) ## visualize raw dataset in the 1st column
                if row_ind == 0:
                    if i == 0:
                        ax.set_title('One-shot', fontsize=self.title_fontsize)
                    else:
                        ax.set_title('Step %d' % i, fontsize=self.title_fontsize)
                col_ind += 1
            ax = fig.add_subplot(gs[row_ind, col_ind])
            self.Plot_onestep(ax, recon_list[-1])
            if row_ind == 0:
                ax.set_title('Reconstruction', fontsize=self.title_fontsize)

    def Plot_metrics(self, elbo_lists, ess_lists, sample_size):
        num_cols = len(elbo_lists)
        gs = gridspec.GridSpec(2, num_cols)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.2, hspace=0.15)
        fig = plt.figure(figsize=(self.fs, self.fs * 2 / num_cols))
        for col_ind in range(num_cols):
            ax1 = fig.add_subplot(gs[0, col_ind])
            ax2 = fig.add_subplot(gs[1, col_ind])
            ax1.plot(elbo_lists[col_ind].data.numpy(), c=self.colors[0])
            ax2.plot(ess_lists[col_ind].data.numpy(), c=self.colors[-1])
            ax1.set_title('Dataset %d' % (col_ind+1), fontsize=self.title_fontsize)
            if col_ind == 0:
                ax1.set_ylabel('ELBO', fontsize=self.title_fontsize)
                ax2.set_ylabel('ESS (L = %d)' % sample_size, fontsize=self.title_fontsize)
