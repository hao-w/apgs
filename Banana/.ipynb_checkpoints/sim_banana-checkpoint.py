import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal


class Sim_BANANA():
    """
    N : number of points
    D : data point dim
    alpha, beta, mu, nu are parameters of normal-gamma
    """
    def __init__(self, N, D, alpha, beta, mu, nu):
        super().__init__()

        self.N = N
        self.D = D
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu = nu

    def sim_one_banana(self):
        precision = Gamma(torch.ones(self.D) * self.alpha, torch.ones(self.D) * self.beta).sample()
        sigma_of_mean = 1. / (precision * self.nu).sqrt()
        sigma = 1. / torch.sqrt(precision)
        mean = Normal(torch.ones(self.D) * self.mu, sigma_of_mean).sample()
        z = Normal(mean, sigma).sample((self.N, ))
        ob = torch.zeros((self.N, self.D))
        ob[:, 0] = z[:, 0]
        ob[:, 1] = z[:, 1] + z[:, 0]**2 + 1
        return ob.data.numpy(), precision.data.numpy(), mean.data.numpy()

    def viz_data(self, num_seqs=20, bound=15, fs=6, colors=['#AA3377', '#EE7733', '#0077BB', '#009988', '#555555', '#999933']):
        for s in range(num_seqs):
            ob, precision, mean = self.sim_one_banana()
            fig, ax = plt.subplots(figsize=(fs, fs))
            ax.scatter(ob[:, 0], ob[:, 1])
#             ax.set_ylim([-bound, bound])
#             ax.set_xlim([-bound, bound])
            ax.set_xticks([])
            ax.set_yticks([])

    def sim_save_data(self, num_seqs, PATH):
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        OB = np.zeros((num_seqs, self.N, self.D))
        MEAN = np.zeros((num_seqs, self.D))
        PRECISION = np.zeros((num_seqs, self.D))
        for s in range(num_seqs):
            ob, precision, mean = self.sim_one_banana()
            OB[s] = ob
            MEAN[s] = mean
            PRECISION[s] = precision
        np.save(PATH + 'ob', OB)
        np.save(PATH + 'mean', MEAN)
        np.save(PATH + 'precision', PRECISION)
        
