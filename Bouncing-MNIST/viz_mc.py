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
    def __init__(self, K, viz_interval, FS, TITLE_FS, bound):
        super().__init__()
        self.K = K
        self.viz_interval = viz_interval
        self.FS = FS
        self.TITLE_FS = TITLE_FS

    def Plot_samples(self, Metrics, filename):
        ## initialize figure object
        Data = Metrics['data']
        Recons = Metrics['recon']
        NUM_DATASETS = len(Data)
        apg_steps = len(Recons)
        num_cols = Data[0].shape[0]
        num_rows = 2 + int((apg_steps - 1)/ self.viz_interval)
        for g, frames in enumerate(Data):
            gs = gridspec.GridSpec(num_rows, num_cols)
            gs.update(left=0.05 , bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
            fig = plt.figure(figsize=(self.FS, self.FS * num_rows / num_cols))
            for c in range(num_cols):
                ax = fig.add_subplot(gs[0, c])
                ax.imshow(frames[c], cmap='gray')
                if c == 0:
                    ax.set_ylabel('Data', fontsize=self.TITLE_FS)
                ax.set_xticks([])
                ax.set_yticks([])
            row_ind = 1
            for r in range(0, apg_steps, self.viz_interval):
                recon_frames = Recons[r].squeeze(0)[g]
                for inc in range(num_cols):
                    ax = fig.add_subplot(gs[row_ind, inc])
                    ax.imshow(recon_frames[inc], cmap='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if inc == 0 :
                        ax.set_ylabel('Step %d' % r, fontsize=self.TITLE_FS)
                row_ind += 1
            plt.savefig(filename + '-%d-' % g + '.svg', dpi=300)

    def Plot_metrics(self, Metrics, sample_size, filename):
        LL = Metrics['ll']
        ESS = Metrics['ess']
        num_cols = len(log_joint_lists)
        gs = gridspec.GridSpec(2, num_cols)
        gs.update(left=0.05 , bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.05)
        fig = plt.figure(figsize=(self.fs, self.fs * 2 / num_cols))
        plt.rc('axes',edgecolor='#eeeeee')
        for col_ind in range(num_cols):
            ax1 = fig.add_subplot(gs[0, col_ind])
            ax2 = fig.add_subplot(gs[1, col_ind])
            # ax3 = fig.add_subplot(gs[2, col_ind])
            temp = log_joint_lists[col_ind].data.numpy()

            baseline = np.ones(temp.shape[0]) * temp[0]
            ax1.plot(log_joint_lists[col_ind].data.numpy(), c=self.colors[0], marker='o')
            ax1.plot(baseline, '--', c=self.colors[1], alpha=0.4)

            # ax2.plot(elbo_lists[col_ind].data.numpy(), c=self.colors[1], marker='o')
            ax2.plot(ess_lists[col_ind].data.numpy() / sample_size, c=self.colors[2])
            ax2.scatter(np.arange(temp.shape[0]), ess_lists[col_ind].data.numpy() / sample_size, c=self.colors[2], s=6.0)

            # ax1.set_title('N= %d' % ((col_ind+1) * 20), fontsize=self.title_fontsize)
            # if col_ind == 0:
                # ax1.set_ylabel('log p(z, x)', fontsize=self.title_fontsize)
                # ax2.set_ylabel('ELBO', fontsize=self.title_fontsize)
                # ax2.set_ylabel('ESS / L', fontsize=self.title_fontsize)
            ax2.set_ylim([-0.1, 1.1])
            ax1.set_xticks([])
            ax1.set_ylim([temp.min()-50, temp.max()+10])
        plt.savefig(filename + '.svg', dpi=300)
        plt.savefig(filename + '.pdf')
