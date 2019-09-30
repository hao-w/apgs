import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from plots import *

class Rings:
    def __init__(self, num_points, num_clusters, period, bound, center_std, noise_std, radi):
        super().__init__()
        self.num_points = num_points
        self.num_clusters = num_clusters
        self.period = period
        self.bound = bound
        self.center_std = center_std
        self.noise_std = noise_std
        self.radi = radi

    def sim_one_ring(self):
        pts_edge = int(self.num_points / self.period)
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

    def sim_mixture_rings(self, num_seqs):
        ob = []
        state = []
        radi = []
        angle = []
        a = 2.0
        a2 = a**1
        while(True):
            mu = np.random.normal(0, self.center_std, (self.num_clusters, 2))
            if ((mu[0] - mu[1])**2).sum() > a2 and ((mu[0] - mu[2])**2).sum() > a2 and ((mu[1] - mu[2])**2).sum() > a2 and ((mu[0] - mu[3])**2).sum() > a2:
                break
        for k in range(self.num_clusters):
            ob_k, radi_k, angle_k = self.sim_one_ring()
            one_hot_k = np.zeros(self.num_clusters)
            one_hot_k[k] = 1
            ob_k = ob_k + mu[k]
            ob.append(ob_k)
            N = ob_k.shape[0]
            state.append(np.tile(one_hot_k, (N, 1)))
            radi.append(radi_k)
            angle.append(angle_k)
        return np.concatenate(ob, 0), np.concatenate(state, 0), mu, np.concatenate(radi, 0)[:, None], np.concatenate(angle, 0)[:, None]

    def visual_data(self, num_seqs):
        for n in range(num_seqs):
            ob, state, mu, _, _ = self.sim_mixture_rings(num_seqs)
            plot_shapes(ob, state, self.num_clusters, self.bound)

    def sim_save_data(self, num_seqs, path):
        pts_dataset = self.num_points*self.num_clusters
        OB = np.zeros((num_seqs, pts_dataset, 2))
        STATE = np.zeros((num_seqs,  pts_dataset, self.num_clusters))
        MU = np.zeros((num_seqs, self.num_clusters, 2))
        RADI = np.zeros((num_seqs, pts_dataset, 1))
        ANGLE = np.zeros((num_seqs, pts_dataset, 1))

        for n in range(num_seqs):
            ob, state, mu, radi, angle = self.sim_mixture_rings(num_seqs)
            OB[n] = ob
            STATE[n] = state
            MU[n] = mu
            RADI[n] = radi
            ANGLE[n] = angle
        np.save(path + '/ob_%d' % (self.num_points * self.num_clusters), OB)
