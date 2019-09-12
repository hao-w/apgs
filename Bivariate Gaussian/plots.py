import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
# 
# def Plot_updates(updates, bg, sigma_factor=5, pts=1000, fs=10, levels=5):
#     Mu, Sigma = bg.Joint()
#     p_joint = mvn(Mu, Sigma)
#     x1 = np.linspace(bg.mu1.item() - bg.sigma1.item() * sigma_factor, bg.mu1.item() + bg.sigma1.item() * sigma_factor,pts)
#     x2 = np.linspace(bg.mu2.item() - bg.sigma2.item() * sigma_factor, bg.mu2.item() + bg.sigma2.item() * sigma_factor,pts)
#     x1, x2 = np.meshgrid(x1, x2)
#     x1x2 = np.column_stack([x1.flat, x2.flat])
#     pdfs = p_joint.log_prob(torch.Tensor(x1x2)).exp().data.numpy().reshape(x1.shape)
#
#     fig = plt.figure(figsize=(fs, fs))
#     ax = fig.add_subplot(111)
#     ax.contour(x1, x2, pdfs, levels=levels, cmap='Reds')
#     # ax.clabel(cs, inline = 1, fontsize=10)
#     T = updates.shape[0]
#     ax.scatter(updates[:, 0].data.numpy(), updates[:, 1].data.numpy(), c=T - np.arange(T))
#     ax.plot(updates[:, 0].data.numpy(), updates[:, 1].data.numpy(), linewidth=1.0, c='k', alpha=0.3)
def Plot_updates(ELBOs, DBs, KLs, updates, bg, sigma_factor=5, pts=1000, fs=10, levels=5, back_to_cpu=True):
    Mu, Sigma = bg.Joint()
    if back_to_cpu:
        Mu = Mu.cpu()
        Sigma = Sigma.cpu()
        updates = updates.cpu()
    p_joint = mvn(Mu, Sigma)
    x1 = np.linspace(bg.mu1.item() - bg.sigma1.item() * sigma_factor, bg.mu1.item() + bg.sigma1.item() * sigma_factor,pts)
    x2 = np.linspace(bg.mu2.item() - bg.sigma2.item() * sigma_factor, bg.mu2.item() + bg.sigma2.item() * sigma_factor,pts)
    x1, x2 = np.meshgrid(x1, x2)
    x1x2 = np.column_stack([x1.flat, x2.flat])
    pdfs = p_joint.log_prob(torch.Tensor(x1x2)).exp().data.numpy().reshape(x1.shape)

    fig = plt.figure(figsize=(fs*4, fs))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.contour(x1, x2, pdfs, levels=levels, cmap='Reds')
    # ax.clabel(cs, inline = 1, fontsize=10)
    T = updates.shape[0]
    ax1.scatter(updates[:, 0].data.numpy(), updates[:, 1].data.numpy(), c=T - np.arange(T))
    ax1.plot(updates[:, 0].data.numpy(), updates[:, 1].data.numpy(), linewidth=1.0, c='k', alpha=0.3)
    ax2 = fig.add_subplot(1, 4, 2)
    for i, db in enumerate(DBs):
        ax2.plot(db.cpu().data.numpy(), label='q_x%d' % (i+1))
    ax2.legend()
    ax2.set_title('Detailed Balance')
    ax3 = fig.add_subplot(1, 4, 3)
    for i, kl in enumerate(KLs):
        ax3.plot(kl.cpu().data.numpy(), label='q_x%d' % (1+i))
    ax3.legend()
    ax3.set_title('KL(P || Q)')
    
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(ELBOs.cpu().data.numpy(), 'o-', label='ELBOs')
    ax4.legend()