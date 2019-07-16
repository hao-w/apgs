import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.gridspec as gridspec
import math
from utils import shuffler, resample

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

def plot_rings(obs, states, K, bound):
    colors = ['r', 'b', 'g', 'k', 'y']
    assignments = states.argmax(-1)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    for k in range(K):
        obs_k = obs[np.where(assignments==k)]
        ax.scatter(obs_k[:,0], obs_k[:, 1], s=5, alpha=0.8)
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])

def plot_final_samples(ob, mu, state, K, PATH):
    page_width = 25
    B, N, D = ob.shape
    plt.rc('axes',edgecolor='#eeeeee')
    gs = gridspec.GridSpec(int(B / 5), 5)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(page_width,page_width*4/5))
    colors = ['#EE7733', 'm', '#0077BB', '#009988']
    for b in range(B):
        ax = fig.add_subplot(gs[int(b/5), int(b%5)])
        assignments = state[b].argmax(-1)
        for k in range(K):
            ob_k = ob[b][np.where(assignments == k)]
            ax.scatter(ob_k[:, 0], ob_k[:, 1], c=colors[k], s=6, alpha=0.8)
            ax.set_ylim([-10, 10])
            ax.set_xlim([-10, 10])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.scatter(mu[b, k, 0], mu[b, k, 1], c=colors[k], marker='X')
#     plt.savefig('../results/modes-' + PATH + '.svg')

def plot_one_step(ob, state, mu, step, fig, gs, opacity, mu_marker, mu_marker_size, colors, title, data_flag=False):
    B, N, D = ob.shape
    for b in range(B):
        ax = fig.add_subplot(gs[b, step])
        xb = ob[b]
        if data_flag:
            ax.scatter(xb[:, 0], xb[:, 1], c='k', s=3.0, alpha=opacity, zorder=3)
        else:
            K = state.shape[-1]
            zb = state[b]
            mub = mu[b].reshape(K, D)
            assignments = zb.argmax(-1)
            for k in range(K):
                xk = xb[np.where(assignments == k)]
                ax.scatter(xk[:, 0], xk[:, 1], c=colors[k], s=3.0, alpha=opacity, zorder=3)
                ax.scatter(mub[k, 0], mub[k, 1], marker=mu_marker, s= mu_marker_size, c=colors[k])
        ax.set_ylim([-10, 10])
        ax.set_xlim([-10, 10])
        ax.set_xticks([])
        ax.set_yticks([])
        if b == 0:
            ax.set_title(title, fontsize=20)

def plot_mc(models, Data, K, data_ptr, mcmc_steps, vis_interval, page_width, CUDA, device):
    sample_size = 1
    batch_size = 5
    marker = 'X'
    marker_size = 100
    opacity = 0.8
    plt.rc('axes',edgecolor='#eeeeee')
    colors = ['#EE7733', 'm', '#0077BB', '#009988']
    gs = gridspec.GridSpec(batch_size, 2+int(mcmc_steps / vis_interval))
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(page_width,page_width*5/7))
    num_datasets = Data.shape[0]
    indices = torch.arange(num_datasets)
    batch_indices = indices[data_ptr*batch_size : (data_ptr+1)*batch_size]
    ob = Data[batch_indices]
    ob = shuffler(ob).repeat(sample_size, 1, 1, 1)
    if CUDA:
        with torch.cuda.device(device):
            ob =ob.cuda()
    test_ob = ob[0].cpu().data.numpy()
    (oneshot_mu, oneshot_state, enc_angle, enc_mu, enc_state, dec_x) = models
    plot_one_step(test_ob, [], [], 0, fig, gs, opacity, marker, marker_size, colors, 'Data', data_flag=True)
    state, angle, mu, w, eubo, elbo, q_mu, q_state = oneshot(oneshot_mu, oneshot_state, enc_angle, dec_x, ob, K)
    E_state = q_state['zs'].dist.probs[0].cpu().data.numpy()
    E_mu = q_mu['means'].dist.loc[0].cpu().data.numpy()
    plot_one_step(test_ob, E_state, E_mu, 1, fig, gs, opacity, marker, marker_size, colors, 'One-shot')
    for m in range(mcmc_steps):
        if m == 0:
            state = resample(state, w, idw_flag=False) ## resample state
            angle = resample(angle, w, idw_flag=False)
        else:
            angle = resample(angle, w_angle, idw_flag=True)
        ## update mu
        mu, w_mu, eubo_mu, elbo_mu, q_mu, _  = Update_mu(enc_mu, dec_x, ob, state, angle, mu, K)
        mu = resample(mu, w_mu, idw_flag=True)
        ## update z
        state, w_state, eubo_state, elbo_state, q_state, _ = Update_state(enc_state, dec_x, ob, angle, mu, state)
        state = resample(state, w_state, idw_flag=True)
        ##update angle
        angle, w_angle, eubo_angle, elbo_angle, _, _ = Update_angle(enc_angle, dec_x, ob, state, mu, angle)
        if (m+1) % vis_interval == 0:
            E_state = q_state['zs'].dist.probs[0].cpu().data.numpy()
            E_mu = q_mu['means'].dist.loc[0].cpu().data.numpy()
            plot_one_step(test_ob, E_state, E_mu, int((m+1) / vis_interval)+1, fig, gs, opacity, marker, marker_size, colors, 'Step %d' % (m+1))
            
def plot_recon(recon, path, page_width, bound):
    S, B, N, D = recon.shape
    gs = gridspec.GridSpec(int(B / 5), 5)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(page_width,page_width*4/5))
    for b in range(B):
        ax = fig.add_subplot(gs[int(b / 5), int(b % 5)])
        ax.scatter(recon[0, b, :, 0], recon[0, b, :, 1], s=5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(-bound, bound)
        ax.set_xlim(-bound, bound)
    plt.savefig('../results/reconstruction-' + path + '.png')
    
def plot_angles(angle_infer, angle_true, page_width):
    S, B, N, D = angle_true.shape
    angle_infer = angle_infer.cpu().data.numpy()
    angle_true = angle_true.cpu().data.numpy()
    gs = gridspec.GridSpec(int(B / 5), 5)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.2, hspace=0.2)
    fig = plt.figure(figsize=(page_width,page_width*4/5))
    for b in range(B):
        ax = fig.add_subplot(gs[int(b / 5), int(b % 5)])
        ax.scatter(angle_true[0, b, :, 0], angle_infer[0, b, :, 0], s=5)
        ax.set_ylim(0, 2*math.pi)
        ax.set_xlim(0, 2*math.pi)