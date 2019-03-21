import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

def plot_trjectory(position, disp, fs=4):
    T = disp.shape[0]
    position = position.data.numpy()
    disp = disp.data.numpy()
    gs = gridspec.GridSpec(2,2)
    fig1 = plt.figure(figsize=(fs,fs))
    ax1 = fig1.gca()
    s = [2*n/4 for n in range(T)]
    ax1.scatter(position[:,0],position[:,1],s=8, c=T - np.arange(T+1))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel('y(t)', fontsize=14)
    ax1.set_xlabel('x(t)', fontsize=14)
    
    tvx = np.concatenate([np.arange(T)[:,None], disp[:,0][:,None]], axis=1)
    tvy = np.concatenate([np.arange(T)[:,None], disp[:,1][:,None]], axis=1)
    vx_seg = np.concatenate([tvx[:-1, None], tvx[1:, None]], axis=1)
    vy_seg = np.concatenate([tvy[:-1, None], tvy[1:, None]], axis=1)
    fig2, (ax2, ax3) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(fs, fs))
    dx_lc = LineCollection(vx_seg)
    dx_lc.set_array(T-2 - np.arange(T-1))
    ax2.add_collection(dx_lc)
    ax2.scatter(np.arange(T), disp[:,0], c=T-1 - np.arange(T), s=4)
    ax2.set_xlabel('t',fontsize=14)
    ax2.set_ylabel(r'$\Delta$x', fontsize=14)
    ax2.set_xticks([])
    ax2.set_yticks([])
    dy_lc = LineCollection(vy_seg)
    dy_lc.set_array(T-2 - np.arange(T-1))
    ax3.add_collection(dy_lc)
    ax3.scatter(np.arange(T), disp[:,1], c=T-1 - np.arange(T), s=4)
    ax3.set_xlabel('t', fontsize=14)
    ax3.set_ylabel(r'$\Delta$y', fontsize=14)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    
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

def plot_clusters(X, mus, covs, K):
    X = X.data.numpy()
    mus = mus.data.numpy()
    covs = covs.data.numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.plot(X[:,0], X[:,1], 'ro')
    for k in range(K):
        plot_cov_ellipse(cov=covs[k], pos=mus[k], nstd=1, ax=ax, alpha=0.5)
    plt.show()

def plot_velocity_circle(v):
    fig = plt.figure(figsize=(4,4))
    ax = fig.gca()
    ax.scatter(v[:,0], v[:,1], label='z=1')
    ax.scatter(-v[:,0], v[:,1], label='z=2')
    ax.scatter(v[:,0], -v[:,1], label='z=3')
    ax.scatter(-v[:,0], -v[:,1], label='z=4')
    ax.set_xlabel('x velocity')
    ax.set_ylabel('y velocity')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)

def plot_transition(As_pred, As_true, vmax, T, num_series, Boundary, signal_noise_ratio, fs):
    As_infer = As_pred / As_pred.sum(-1)[:, :, None]
    As_infer = As_infer.mean(0)
    fig3 = plt.figure(figsize=(fs,fs))
    ax1 = fig3.add_subplot(1, 1, 1)
    infer_plot = ax1.imshow(As_infer, cmap='Greys', vmin=0, vmax=vmax)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('inferred averaged transition matrix')
    ax2 = fig3.add_subplot(2, 1, 2)
    As_true_ave = As_true.mean(0)
    true_plot = ax2.imshow(As_true_ave, cmap='Greys', vmin=0, vmax=vmax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('true averaged transition matrix')
    # cbaxes = fig3.add_axes([0.95, 0.32, 0.02, 0.36])
    # cb = plt.colorbar(true_plot, cax = cbaxes)
    # fig3.savefig('transition_plot T=%d_series=%d_boundary=%d_ratio=%f.png' % (T, num_series, Boundary, signal_noise_ratio))
    return As_infer, As_true_ave

def plot_results_clusters(Mus_true, Mus, Covs, K, num_seqs, fs=8.0):
    fig = plt.figure(figsize=(fs, fs))
    ax1 = fig.add_subplot(1, 1, 1)
    markersize = 10.0
    colors = ['b', 'Purple', 'g', 'r']
    for k in range(K):
        ax1.scatter(Mus[:, k, 0], Mus[: , k, 1], c=colors[k], marker='x')
        ax1.scatter(Mus_true[:, k, 0], Mus_true[:, k, 1], s=markersize, color=colors[k])
        for s in range(num_seqs):
            plot_cov_ellipse(cov=Covs[s, k, :, :], pos=Mus[s, k, :], nstd=0.3, ax=ax1, alpha=0.3)
        
        
def plot_results_transition(As, vmax=1.0, fs=8.0):
    A= As.mean(0)
    fig = plt.figure(figsize=(fs,fs))

    gs2 = gridspec.GridSpec(4, 1)
    gs2.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
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

    ax2.imshow(A[None, 0,:], cmap='Blues', vmin=0, vmax=vmax)
    ax3.imshow(A[None, 1,:], cmap='Purples', vmin=0, vmax=vmax)
    ax4.imshow(A[None, 2,:], cmap='Greens', vmin=0, vmax=vmax)
    ax5.imshow(A[None, 3,:], cmap='Reds', vmin=0, vmax=vmax)