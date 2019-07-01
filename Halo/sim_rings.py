import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from plots import *

class Rings:
    def __init__(self, num_points, num_clusters, period, bound, center_std, noise_std, radi, fixed_radi, collapsed_noise):
        super().__init__()
        self.num_points = num_points
        self.num_clusters = num_clusters
        self.period = period
        self.bound = bound
        self.center_std = center_std
        self.noise_std = noise_std
        self.radi = radi
        self.fixed_radi = fixed_radi
        self.collapsed_noise = collapsed_noise

    def sim_one_ring(self):
        angle_per_point = np.linspace(0, self.period * 2 * math.pi, self.num_points, endpoint=False)

        if self.fixed_radi:
            radi_per_point = np.ones(self.num_points) * self.radi
        else:
            radi_per_point = np.ones(self.num_points) * np.random.uniform(self.radi-0.5, self.radi+0.5, 1)

        if self.collapsed_noise:
            noise = np.random.normal(0.0, self.noise_std, self.num_points)
            radi_per_point += noise
            x = np.cos(angle_per_point) * radi_per_point
            y = np.sin(angle_per_point) * radi_per_point
            pos = np.concatenate((x[:, None], y[:, None]), -1)
        else:
            noise = np.random.normal(0.0, self.noise_std, (self.num_points, 2))
            x = np.cos(angle_per_point) * radi_per_point
            y = np.sin(angle_per_point) * radi_per_point
            pos = np.concatenate((x[:, None], y[:, None]), -1)
            pos = pos + noise
        return pos, radi_per_point, angle_per_point

    def sim_mixture_rings(self, num_seqs):
        ob = []
        state = []
        radi = []
        angle = []
        mu = np.random.normal(0, self.center_std, (self.num_clusters, 2))
        for k in range(self.num_clusters):
            ob_k, radi_k, angle_k = self.sim_one_ring()
            one_hot_k = np.zeros(self.num_clusters)
            one_hot_k[k] = 1
            ob_k = ob_k + mu[k]
            ob.append(ob_k)
            state.append(np.tile(one_hot_k, (self.num_points, 1)))
            radi.append(radi_k)
            angle.append(angle_k)
        return np.concatenate(ob, 0), np.concatenate(state, 0), mu, np.concatenate(radi, 0)[:, None], np.concatenate(angle, 0)[:, None]

    def visual_data(self, num_seqs):
        for n in range(num_seqs):
            ob, state, mu, _, _ = self.sim_mixture_rings(num_seqs)
            plot_rings(ob, state, self.num_clusters, self.bound)

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
        np.save(path + '/ob', OB)
        np.save(path + '/state', STATE)
        np.save(path + '/mu', MU)
        np.save(path + '/radi', RADI)
        np.save(path + '/angle', angle)
