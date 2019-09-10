import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from plots import *

class Rings:
    def __init__(self, num_points, num_clusters, period, bound, center_std, noise_std, radi, fixed_radi, truncate_percent, truncate=False):
        super().__init__()
        self.num_points = num_points
        self.num_clusters = num_clusters
        self.period = period
        self.bound = bound
        self.center_std = center_std
        self.noise_std = noise_std
        self.radi = radi
        self.fixed_radi = fixed_radi
        self.truncate_percent = truncate_percent
        self.truncate = truncate

    def one_missing(self, pts_edge, angles):
        pts_edge_t = int(self.truncate_percent * pts_edge)
        pts_edge_r = pts_edge - pts_edge_t
        starting_pt = np.random.choice(pts_edge)
        if pts_edge - starting_pt < pts_edge_r:
            diff = starting_pt + pts_edge_r - pts_edge
            angles = angles[diff:starting_pt]
        else:
            angles = np.concatenate((angles[:starting_pt], angles[starting_pt+pts_edge_r:]), 0)
        return angles

    def two_missing(self, pts_edge, angles):
        pts_edge_t = int(self.truncate_percent * pts_edge)
        pts_edge_r = pts_edge - pts_edge_t
        half = int(pts_edge/2)
        starting_pt_1 = np.random.choice(half)
        starting_pt_2 = starting_pt_1 + half

        if half < starting_pt_1 + pts_edge_r/2:
            diff = starting_pt_1 + int(pts_edge_r/2) - half
            angles = np.concatenate((angles[diff:starting_pt_1], angles[half+diff : starting_pt_2]), 0)
        else:
            angles = np.concatenate((angles[:starting_pt_1], angles[starting_pt_1+int(pts_edge_r/2):half], angles[half:starting_pt_2], angles[starting_pt_2+int(pts_edge_r/2):]), 0)
        return angles

    def three_missing(self, pts_edge, angles):
        pts_edge_t = int(self.truncate_percent * pts_edge)
        pts_edge_r = pts_edge - pts_edge_t
        one_thirds = int(pts_edge/3)
        starting_pt_1 = np.random.choice(one_thirds)
        starting_pt_2 = starting_pt_1 + one_thirds
        starting_pt_3 = starting_pt_2 + one_thirds

        if one_thirds < starting_pt_1 + pts_edge_r/3:
            diff = starting_pt_1 + int(pts_edge_r/3) - one_thirds
            angles = np.concatenate((angles[diff:starting_pt_1], angles[one_thirds+diff : starting_pt_2], angles[2*one_thirds+diff : starting_pt_3]), 0)
        else:
            angles = np.concatenate((angles[:starting_pt_1], angles[starting_pt_1+int(pts_edge_r/3):one_thirds], angles[one_thirds:starting_pt_2], angles[starting_pt_2+int(pts_edge_r/3):2*one_thirds], angles[2*one_thirds : starting_pt_3], angles[starting_pt_3+int(pts_edge_r/3) : ]), 0)
        return angles

    def four_missing(self, pts_edge, angles):
        pts_edge_t = int(self.truncate_percent * pts_edge)
        pts_edge_r = pts_edge - pts_edge_t
        one_fourths = int(pts_edge/4)
        starting_pt_1 = np.random.choice(one_fourths)
        starting_pt_2 = starting_pt_1 + one_fourths
        starting_pt_3 = starting_pt_2 + one_fourths
        starting_pt_4 = starting_pt_3 + one_fourths
        if one_fourths < starting_pt_1 + pts_edge_r/4:
            diff = starting_pt_1 + int(pts_edge_r/4) - one_fourths
            angles = np.concatenate((angles[diff:starting_pt_1], angles[one_fourths+diff : starting_pt_2], angles[2*one_fourths+diff : starting_pt_3], angles[3*one_fourths+diff : starting_pt_4]), 0)
        else:
            angles = np.concatenate((angles[:starting_pt_1], angles[starting_pt_1+int(pts_edge_r/4):one_fourths], angles[one_fourths:starting_pt_2], angles[starting_pt_2+int(pts_edge_r/4):2*one_fourths], angles[2*one_fourths : starting_pt_3], angles[starting_pt_3+int(pts_edge_r/4) : 3*one_fourths], angles[3*one_fourths : starting_pt_4], angles[starting_pt_4+int(pts_edge_r/4) : ]), 0)
        return angles

    def sim_one_ring(self):
        pts_edge = int(self.num_points / self.period)
        angles = np.linspace(0, 2 * math.pi, pts_edge, endpoint=True)
        if self.truncate:
            raffle = np.random.choice(3)
            if raffle == 0:
                angles = self.two_missing(pts_edge, angles)
            elif raffle == 1:
                angles = self.three_missing(pts_edge, angles)
            elif raffle == 2:
                angles = self.four_missing(pts_edge, angles)
            else:
                angles = self.one_missing(pts_edge, angles)
        angles = np.tile(angles, self.period)
        N = angles.shape[0]
        if self.fixed_radi:
            radis = np.ones(N) * self.radi

        else:
            radis= np.ones(N) * np.random.uniform(self.radi-0.5, self.radi+0.5, 1)
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
        mu = np.random.normal(0, self.center_std, (self.num_clusters, 2))
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
            plot_rings(ob, state, self.num_clusters, self.bound)

    def sim_save_data(self, num_seqs, path):
        if self.truncate:
            N = int(self.num_points * self.truncate_percent)
            pts_dataset = N*self.num_clusters
            OB = np.zeros((num_seqs, pts_dataset, 2))
            STATE = np.zeros((num_seqs,  pts_dataset, self.num_clusters))
            MU = np.zeros((num_seqs, self.num_clusters, 2))
            RADI = np.zeros((num_seqs, pts_dataset, 1))
            ANGLE = np.zeros((num_seqs, pts_dataset, 1))
        else:
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

