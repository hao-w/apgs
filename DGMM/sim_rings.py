import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

class Sim_Rings():
    """
    N : number of points in one dataset
    K : number of clusters
    Nk = numer of points in one cluster
    D : data point dim
    mu_std : standard deviation of the cluster center
    noise_std : standard deviation of the 2D gaussian noise
    radi : radius of a ring (fixed from ring to ring)
    """
    def __init__(self, N, K, D, period, mu_std, noise_std, radi):
        super().__init__()
        self.N = N
        self.K = K
        self.D = D
        self.Nk = int(self.N / self.K)
        self.period = period
        # self.bound = bound
        self.mu_std = mu_std
        self.noise_std = noise_std
        self.radi = radi

    def sim_one_ring(self):
        pts_edge = int(self.Nk / self.period)
        angles = np.linspace(0, 2 * math.pi, pts_edge, endpoint=True)
        angles = np.tile(angles, self.period)
        N = angles.shape[0]
        radis = np.ones(N) * self.radi
        noise = np.random.normal(0.0, self.noise_std, (N, 2))
        x = np.cos(angles) * radis
        y = np.sin(angles) * radis
        pos = np.concatenate((x[:, None], y[:, None]), -1)
        pos = pos + noise
        return pos, radis, angles

    def sim_one_ncmm(self):
        ob = []
        state = []
        radi = []
        angle = []
        mu = np.random.normal(0, self.mu_std, (self.K, 2))
        # helped to generate less overlapped rings
        # the apg sampler does not depend on this assumption
        # just for better visualization effect in the paper
        a = 2.0
        a2 = a**1
        while(True):
            if(
               ((mu[0] - mu[1])**2).sum() > a2 and
               ((mu[0] - mu[2])**2).sum() > a2 and
               ((mu[0] - mu[3])**2).sum() > a2 and
               ((mu[1] - mu[2])**2).sum() > a2 and
               ((mu[1] - mu[3])**2).sum() > a2 and
               ((mu[2] - mu[3])**2).sum() > a2
               ):
                break
            mu = np.random.normal(0, self.mu_std, (self.K, 2))
        for k in range(self.K):
            ob_k, radi_k, angle_k = self.sim_one_ring()
            one_hot_k = np.zeros(self.K)
            one_hot_k[k] = 1
            ob_k = ob_k + mu[k]
            ob.append(ob_k)
            N = ob_k.shape[0]
            state.append(np.tile(one_hot_k, (N, 1)))
            radi.append(radi_k)
            angle.append(angle_k)
        return np.concatenate(ob, 0), np.concatenate(state, 0), mu, np.concatenate(radi, 0)[:, None], np.concatenate(angle, 0)[:, None]

    def viz_data(self, num_seqs=20, bound=10, fs=6, colors=['#AA3377', '#EE7733', '#0077BB', '#009988', '#555555', '#999933']):
        for s in range(num_seqs):
            ob, state, mu, _, _ = self.sim_one_ncmm()
            assignments = state.argmax(-1)
            fig, ax = plt.subplots(figsize=(fs, fs))
            for k in range(self.K):
                ob_k = ob[np.where(assignments==k)]
                ax.scatter(ob_k[:,0], ob_k[:, 1], c=colors[k], s=5, alpha=0.8)
            ax.set_xlim([-bound, bound])
            ax.set_ylim([-bound, bound])
            ax.set_xticks([])
            ax.set_yticks([])

    def sim_save_data(self, num_seqs, PATH):
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        OB = np.zeros((num_seqs, self.N, self.D))
        # STATE = np.zeros((num_seqs,  pts_dataset, self.num_clusters))
        # MU = np.zeros((num_seqs, self.num_clusters, 2))
        # RADI = np.zeros((num_seqs, pts_dataset, 1))
        # ANGLE = np.zeros((num_seqs, pts_dataset, 1))

        for s in range(num_seqs):
            ob, state, mu, radi, angle = self.sim_one_ncmm()
            OB[s] = ob
        np.save(PATH + 'ob', OB)
