import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from plots import *

class Squares:
    def __init__(self, num_points, num_clusters, period, bound, center_std, noise_std, radi, fixed_radi):
        super().__init__()
        self.num_points = num_points
        self.num_clusters = num_clusters
        self.period = period
        self.bound = bound
        self.center_std = center_std
        self.noise_std = noise_std
        self.radi = radi
        self.fixed_radi = fixed_radi

    def sim_one_square(self):
        pts_edge = int(self.num_points / self.period / 4)
        upper_x = np.linspace(-self.radi, self.radi, pts_edge, endpoint=False)
        upper_y = np.ones(pts_edge) * self.radi
        lower_x = np.linspace(-self.radi, self.radi, pts_edge, endpoint=False)
        lower_y = np.ones(pts_edge) * (-self.radi)
        left_y = np.linspace(-self.radi, self.radi, pts_edge, endpoint=False)
        left_x = np.ones(pts_edge) * (-self.radi)
        right_y = np.linspace(-self.radi, self.radi, pts_edge, endpoint=False)
        right_x = np.ones(pts_edge) * (self.radi)
        x = np.concatenate((upper_x, right_x, lower_x, left_x), 0)
        y = np.concatenate((upper_y, right_y, lower_y, left_y), 0)
        pos = np.concatenate((x[:, None], y[:, None]), -1)
        pos = np.tile(pos, (self.period, 1))
        rads = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        angles = np.arctan2(pos[:, 1], pos[:, 0])

        noise = np.random.normal(0.0, self.noise_std, (self.num_points, 2))

        pos = pos + noise
        return pos, rads, angles

    def sim_mixture_squares(self, num_seqs):
        ob = []
        state = []
        radi = []
        angle = []
        mu = np.random.normal(0, self.center_std, (self.num_clusters, 2))
        for k in range(self.num_clusters):
            ob_k, radi_k, angle_k = self.sim_one_square()
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
            ob, state, mu, _, _ = self.sim_mixture_squares(num_seqs)
            plot_shapes(ob, state, self.num_clusters, self.bound)

    def sim_save_data(self, num_seqs, path):
        pts_dataset = self.num_points*self.num_clusters
        OB = np.zeros((num_seqs, pts_dataset, 2))
        STATE = np.zeros((num_seqs,  pts_dataset, self.num_clusters))
        MU = np.zeros((num_seqs, self.num_clusters, 2))
        RADI = np.zeros((num_seqs, pts_dataset, 1))
        ANGLE = np.zeros((num_seqs, pts_dataset, 1))
        for n in range(num_seqs):
            ob, state, mu, radi, angle = self.sim_mixture_squares(num_seqs)
            OB[n] = ob
            STATE[n] = state
            MU[n] = mu
            RADI[n] = radi
            ANGLE[n] = angle
        np.save(path + '/ob_%d' % (self.num_points * self.num_clusters), OB)
        np.save(path + '/state_%d' % (self.num_points * self.num_clusters), STATE)
        np.save(path + '/mu_%d' % (self.num_points * self.num_clusters), MU)
        np.save(path + '/angle_%d' % (self.num_points * self.num_clusters), ANGLE)
