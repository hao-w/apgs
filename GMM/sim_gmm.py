import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from viz import plot_cov_ellipse
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat


class Sim_GMM():
    """
    N : number of points
    K : number of clusters
    D : data point dim
    alpha, beta, mu, nu are parameters of normal-gamma
    """
    def __init__(self, N, K, D, alpha, beta, mu, nu):
        super().__init__()

        self.N = N
        self.K = K
        self.D = D
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu = nu

    def sim_one_gmm(self):
        precision = Gamma(torch.ones((self.K, self.D)) * self.alpha, torch.ones((self.K, self.D)) * self.beta).sample()
        sigma = 1. / torch.sqrt(precision)
        sigma_of_mean = sigma / self.nu ## nu
        mean = Normal(torch.ones((self.K, self.D)) * self.mu, sigma_of_mean).sample()
        assignment = cat(torch.ones(self.K) * (1. / self.K)).sample((self.N,))
        labels = assignment.nonzero()[:, 1]
        ob = Normal(mean[labels], sigma[labels]).sample()
        return ob.data.numpy(), precision.data.numpy(), mean.data.numpy(), assignment.data.numpy()

    def viz_data(self, num_seqs=20, bound=15, fs=6, colors=['#AA3377', '#EE7733', '#0077BB', '#009988', '#555555', '#999933']):
        for s in range(num_seqs):
            ob, precision, mean, assignment = self.sim_one_gmm()
            fig, ax = plt.subplots(figsize=(fs, fs))
            labels = assignment.argmax(-1)
            for k in range(self.K):
                cov_k = np.diag(1. / precision[k])
                ob_k = ob[np.where(labels == k)]
                ax.scatter(ob_k[:, 0], ob_k[:, 1], c=colors[k], zorder=3)
                plot_cov_ellipse(cov=cov_k, pos=mean[k], nstd=2, color=colors[k], ax=ax, alpha=0.3, zorder=3)
            ax.set_ylim([-bound, bound])
            ax.set_xlim([-bound, bound])
            ax.set_xticks([])
            ax.set_yticks([])

    def sim_save_data(self, num_seqs, PATH):
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        OB = np.zeros((num_seqs, self.N, self.D))
        # ASSIGNMENT = np.zeros((num_seqs, self.N. self.K))
        # MEAN = np.zeros((num_seqs, self.K, self.D))
        # SIGMA = np.zeros((num_seqs, self.K, self.D))
        for s in range(num_seqs):
            ob, precision, mean, assignment = self.sim_one_gmm()
            OB[s] = ob
        np.save(PATH + 'ob', OB)
