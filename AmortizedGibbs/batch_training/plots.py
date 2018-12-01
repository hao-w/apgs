import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.gridspec as gridspec

def pairwise(Zs, T):
    return torch.bmm(Zs[:T-1].unsqueeze(-1), Zs[1:].unsqueeze(1))

def plot_dirs(variational, alpha_trans_0, Zs, T, K, vmax):
    conjugate_post = alpha_trans_0 + pairwise(Zs, T).sum(0)
    print('variational : ')
    print(variational)
    print('conjugate posterior :')
    print(conjugate_post)

    fig3 = plt.figure(figsize=(12,6))
    ax1 = fig3.add_subplot(1, 2, 1)
    infer_plot = ax1.imshow(variational.data.numpy(), cmap='viridis', vmin=0, vmax=vmax)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('variational')
    ax2 = fig3.add_subplot(1, 2, 2)
    true_plot = ax2.imshow(conjugate_post.data.numpy(), cmap='viridis', vmin=0, vmax=vmax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('conjugate posterior')
    cax = fig3.add_axes([1.0, 0.15, 0.03, 0.7])
    #fig3.colorbar(true_plot, cax=cax, orientation='vertical')
    # cbaxes = fig3.add_axes([0.95, 0.32, 0.02, 0.36])
    # cb = plt.colorbar(true_plot, cax = cbaxes)
    # fig3.savefig('transition_plot T=%d_series=%d_boundary=%d_ratio=%f.png' % (T, num_series, Boundary, signal_noise_ratio))

def plot_results(EUBOs, ELBOs, ESSs, KLs, filename):
    fig, ax = plt.subplots(figsize=(8,24))
    ax.set_xticks([])
    ax.set_yticks([])
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(EUBOs, 'r-', label='eubo')
    ax1.plot(ELBOs, 'b-', label='elbo')
    ax1.legend(fontsize=18)
    ax1.set_xlabel('gradient steps', fontsize=18)
    ax1.set_ylabel('eubo and elbo estimators', fontsize=18)
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(KLs, 'g-', label='true kl')
    ax2.plot(np.array(EUBOs) - np.array(ELBOs), 'k-', label='bounds gap')
    ax2.set_xlabel('gradient steps', fontsize=18)
    ax2.set_ylabel('KL(p(eta | z, y) || q_phi (\eta | z))', fontsize=18)
    ax2.legend(fontsize=18)
    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(ESSs, 'm-o')
    ax3.set_xlabel('gradient steps', fontsize=18)
    ax3.set_ylabel('effective sample sizes', fontsize=18)
    ax3.set_ylim([1, 2])
    plt.savefig(filename)

def plot_smc_sample(Zs_true, Zs_ret):
    ret_index = torch.nonzero(Zs_ret).data.numpy()
    true_index = torch.nonzero(Zs_true).data.numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(true_index[:,0], true_index[:,1], 'ro', label='truth')
    ax.plot(ret_index[:,0], ret_index[:,1], 'bo', label='sample')
    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 0.1))
    plt.show()

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

def plot_clusters(Xs, mus, covs, K):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axis('equal')
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    for k in range(K):
        plot_cov_ellipse(cov=covs[k], pos=mus[k], nstd=2, ax=ax, alpha=0.5)
    plt.show()

def plot_circle_transition_colorcode(num_series, init_v, final_mus, final_covs, As_pred, As_true, K, fs, vmax, width_space, height_space, cov_flag, legend_flag, save_flag):

    As_infer = As_pred / As_pred.sum(-1)[:, :, None]
    As_infer = As_infer.mean(0)
    As_true_ave = As_true.mean(0)

    fig = plt.figure(figsize=(fs*1.5 + width_space,fs + height_space))
    gs1 = gridspec.GridSpec(1, 1)
    # , width_ratios=[2,1], height_ratios=[1,1]
    gs1.update(left=0.0, bottom=0.0, right=(2/3), top=1.0, wspace=width_space, hspace=height_space)
    ax1 = fig.add_subplot(gs1[0])

    # ax3 = fig.add_subplot(gs[1, 1])
    ax1.set_xticks([])
    ax1.set_yticks([])
## increment weights
    gs2 = gridspec.GridSpec(4, 1)
    gs2.update(left=2/3 + (1/3)*width_space, bottom=0.5+(1/2)*width_space, right=1.0, top=1.0, wspace=0, hspace=0)
    ax2 = fig.add_subplot(gs2[0, 0])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3 = fig.add_subplot(gs2[1, 0])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4 = fig.add_subplot(gs2[2, 0])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5 = fig.add_subplot(gs2[3, 0])
    ax5.set_xticks([])
    ax5.set_yticks([])

    gs3 = gridspec.GridSpec(4, 1)
    gs3.update(left=2/3 + (1/3)*width_space, bottom=0.0, right=1.0, top=0.5 - (1/2)*width_space, wspace=0, hspace=0)
    ax6 = fig.add_subplot(gs3[0, 0])
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax7 = fig.add_subplot(gs3[1, 0])
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax8 = fig.add_subplot(gs3[2, 0])
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax9 = fig.add_subplot(gs3[3, 0])
    ax9.set_xticks([])
    ax9.set_yticks([])
    markersize = 10.0
    colors = ['b', 'Purple', 'g', 'r']
    ## plot left one
    ax1.scatter(init_v[:,0], init_v[:,1], s=markersize, color=colors[0], label='z=1')
    ax1.scatter(init_v[:,0], -init_v[:,1], s=markersize, color=colors[1], label='z=2')
    ax1.scatter(-init_v[:,0], -init_v[:,1], s=markersize, color=colors[2], label='z=3')
    ax1.scatter(-init_v[:,0], init_v[:,1], s=markersize, color=colors[3], label='z=4')

    for k in range(K):
        ax1.scatter(final_mus[:,k,0], final_mus[:,k,1], c=colors[k], marker='x')
    if cov_flag:
        for k in range(K):
            for s in range(num_series):
                plot_cov_ellipse(cov=final_covs[s, k, :, :], pos=final_mus[s, k, :], nstd=0.3, ax=ax1, alpha=0.3)
    #    ax1.set_xlabel('x velocity')
    #    ax1.set_ylabel('y velocity')
    if legend_flag:
        ax1.legend(loc='upper center', bbox_to_anchor=(0.75, 1.15), ncol=4)

    ax2.imshow(As_infer[None, 0,:], cmap='Blues', vmin=0, vmax=vmax)
    ax3.imshow(As_infer[None, 1,:], cmap='Purples', vmin=0, vmax=vmax)
    ax4.imshow(As_infer[None, 2,:], cmap='Greens', vmin=0, vmax=vmax)
    ax5.imshow(As_infer[None, 3,:], cmap='Reds', vmin=0, vmax=vmax)

    ax6.imshow(As_true_ave[None, 0,:], cmap='Blues', vmin=0, vmax=vmax)
    ax7.imshow(As_true_ave[None, 1,:], cmap='Purples', vmin=0, vmax=vmax)
    ax8.imshow(As_true_ave[None, 2,:], cmap='Greens', vmin=0, vmax=vmax)
    ax9.imshow(As_true_ave[None, 3,:], cmap='Reds', vmin=0, vmax=vmax)

    if save_flag:
        fig.savefig('baseline_results_1003.pdf', bbox_inches='tight')
        fig.savefig('baseline_results_1003.svg', bbox_inches='tight')
        fig.savefig('baseline_results_1003.png', dpi=600, bbox_inches='tight')
