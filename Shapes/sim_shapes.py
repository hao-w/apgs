import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from plots import *

class Shapes:
    def __init__(self, pts, num_clusters, period, bound, center_std, noise_std, radi):
        super().__init__()
        (self.pts_c, self.pts_s, self.pts_t) = pts
        self.num_clusters = num_clusters
        self.period = period
        self.bound = bound
        self.center_std = center_std
        self.noise_std = noise_std
        self.radi = radi

    def sim_one_square(self):
        pts_edge = int(self.pts_s / self.period / 4)
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
        radi = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        angle = np.arctan2(pos[:, 1], pos[:, 0])
        noise = np.random.normal(0.0, self.noise_std, (self.pts_s, 2))
        pos = pos + noise
        return pos, radi, angle

    def sim_one_triangle(self):
        pts_edge = int(self.pts_t/ self.period / 3)
        lower_x = np.linspace(-self.radi, self.radi, pts_edge, endpoint=False)
        lower_y = np.zeros(pts_edge) - 0.5 * self.radi
        lower = np.concatenate((lower_x[:, None], lower_y[:, None]), -1)
        left_x = np.linspace(-self.radi, self.radi, pts_edge, endpoint=False)
        left_y = np.zeros(pts_edge) + 0.5 * self.radi
        right_x = np.linspace(-self.radi, self.radi, pts_edge, endpoint=False)
        right_y = np.zeros(pts_edge) + 0.5 * self.radi
        rotation_left = np.pi * 2 / 3
        rotation_matrix_left = np.array([[np.cos(rotation_left), -np.sin(rotation_left)], [np.sin(rotation_left), np.cos(rotation_left)]])
        rotation_right = np.pi / 3
        rotation_matrix_right = np.array([[np.cos(rotation_right), -np.sin(rotation_right)], [np.sin(rotation_right), np.cos(rotation_right)]])
        left = np.concatenate((left_x[:, None], left_y[:, None]), -1)
        right = np.concatenate((right_x[:, None], right_y[:, None]), -1)
        left = np.matmul(left, rotation_matrix_left)
        left = left + np.array([-2, 1])
        offset_y = np.sqrt(3) * self.radi - 1.0 - left[0, 1]
        offset_x = -self.radi - left[-1, 0]
        right = np.matmul(right, rotation_matrix_right)
        left[:, 1] = left[:, 1] + offset_y
        right[:, 1] = right[:, 1] + offset_y
        left[:, 0] = left[:, 0] + offset_x
        right[:, 0] = right[:, 0] + offset_x
        pos = np.concatenate((lower, left, right), 0)
        pos = np.tile(pos, (self.period, 1))
        radi = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        angle = np.arctan2(pos[:, 1], pos[:, 0])
        noise = np.random.normal(0.0, self.noise_std, (self.pts_t, 2))
        pos = pos + noise
        return pos, radi, angle

    def sim_one_ring(self):
        pts_edge = int(self.pts_c / self.period)
        angle = np.linspace(0, 2 * math.pi, pts_edge, endpoint=True)
        angle = np.tile(angle, self.period)
        N = angle.shape[0]
        radi = np.ones(N) * self.radi
        noise = np.random.normal(0.0, self.noise_std, (N, 2))
        x = np.cos(angle) * radi
        y = np.sin(angle) * radi
        pos = np.concatenate((x[:, None], y[:, None]), -1)
        pos = pos + noise
        return pos, radi, angle

    def sim_mixture_shapes(self, num_seqs):
        state = []
        radi = []
        angle = []
        # alphac = np.random.uniform(0, 2/3 * math.pi, 1)
        # radic = np.random.uniform(3.0, 10.0, 1)
        #
        # alphas = np.random.uniform(2/3 * math.pi, 4/3 * math.pi, 1)
        # radis = np.random.uniform(3.0, 8.0, 1)
        #
        # alphat = np.random.uniform(4/3 * math.pi, 2 * math.pi, 1)
        # radit = np.random.uniform(3.0, 8.0, 1)
        #
        # mu = np.zeros((self.num_clusters, 2))
        # mu[0, 0] = np.cos(alphac) * radic
        # mu[0, 1] = np.sin(alphac) * radic
        #
        # mu[1, 0] = np.cos(alphas) * radis
        # mu[1, 1] = np.sin(alphas) * radis
        #
        # mu[2, 0] = np.cos(alphat) * radit
        # mu[2, 1] = np.sin(alphat) * radit
        a = 2.0
        a2 = a**2
        while(True):
            mu = np.random.normal(0, self.center_std, (self.num_clusters, 2))
            if ((mu[0] - mu[1])**2).sum() > a2 and ((mu[0] - mu[2])**2).sum() > a2 and ((mu[1] - mu[2])**2).sum() > a2 and ((mu[0] - mu[3])**2).sum() > a2:
                break
        ob_c, radi_c, angle_c = self.sim_one_ring()
        ob_s, radi_s, angle_s = self.sim_one_square()
        ob_t, radi_t, angle_t = self.sim_one_triangle()
        ## shift by variable mu
        ob_c = ob_c + mu[0]
        ob_s = ob_s + mu[1]
        ob_t = ob_t + mu[2]
        state_c = np.tile(np.array([1, 0, 0]), (ob_c.shape[0], 1))
        state_s = np.tile(np.array([0, 1, 0]), (ob_s.shape[0], 1))
        state_t = np.tile(np.array([0, 0, 1]), (ob_t.shape[0], 1))
        return np.concatenate((ob_c, ob_s, ob_t), 0), np.concatenate((state_c, state_s, state_t), 0), mu, np.concatenate((radi_c, radi_s, radi_t), 0)[:, None], np.concatenate((angle_c, angle_s, angle_t), 0)[:, None]

    def sim_mixture_triangles(self, num_seqs):
        ob = []
        state = []
        radi = []
        angle = []

        a = 2.0
        a2 = a**2
        while(True):
            mu = np.random.normal(0, self.center_std, (self.num_clusters, 2))
            if ((mu[0] - mu[1])**2).sum() > a2 and ((mu[0] - mu[2])**2).sum() > a2 and ((mu[1] - mu[2])**2).sum() > a2 :
                break
        for k in range(self.num_clusters):
            ob_k, radi_k, angle_k = self.sim_one_triangle()
            one_hot_k = np.zeros(self.num_clusters)
            one_hot_k[k] = 1
            ob_k = ob_k + mu[k]
            ob.append(ob_k)
            N = ob_k.shape[0]
            state.append(np.tile(one_hot_k, (N, 1)))
            radi.append(radi_k)
            angle.append(angle_k)
        return np.concatenate(ob, 0), np.concatenate(state, 0), mu, np.concatenate(radi, 0)[:, None], np.concatenate(angle, 0)[:, None]

    def sim_mixture_squares(self, num_seqs):
        ob = []
        state = []
        radi = []
        angle = []

        a = 2.0
        a2 = a**2
        while(True):
            mu = np.random.normal(0, self.center_std, (self.num_clusters, 2))
            if ((mu[0] - mu[1])**2).sum() > a2 and ((mu[0] - mu[2])**2).sum() > a2 and ((mu[1] - mu[2])**2).sum() > a2 :
                break
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


    def visual_data(self, num_seqs, shape):
        if shape == 'shapes':
            for n in range(num_seqs):
                ob, state, mu, _, _ = self.sim_mixture_shapes(num_seqs)
                plot_shapes(ob, state, self.num_clusters, self.bound)
        elif shape == 'squares':
            for n in range(num_seqs):
                ob, state, mu, _, _ = self.sim_mixture_squares(num_seqs)
                plot_shapes(ob, state, self.num_clusters, self.bound)
        elif shape == 'triangles':
            for n in range(num_seqs):
                ob, state, mu, _, _ = self.sim_mixture_triangles(num_seqs)
                plot_shapes(ob, state, self.num_clusters, self.bound)
        else:
            print('ERROR : shape name undefined!')

    def sim_save_data(self, num_seqs, path, shape):
        if shape == 'shapes':
            pts_dataset = self.pts_c + self.pts_s + self.pts_t
            OB = np.zeros((num_seqs, pts_dataset, 2))
            STATE = np.zeros((num_seqs,  pts_dataset, self.num_clusters))
            MU = np.zeros((num_seqs, self.num_clusters, 2))
            RADI = np.zeros((num_seqs, pts_dataset, 1))
            ANGLE = np.zeros((num_seqs, pts_dataset, 1))
            for n in range(num_seqs):
                ob, state, mu, radi, angle = self.sim_mixture_shapes(num_seqs)
                OB[n] = ob
                STATE[n] = state
                MU[n] = mu
                RADI[n] = radi
                ANGLE[n] = angle
        elif shape == 'squares':
            pts_dataset = self.pts_s * self.num_clusters
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
        elif shape == 'triangles':
            pts_dataset = self.pts_t * self.num_clusters
            OB = np.zeros((num_seqs, pts_dataset, 2))
            STATE = np.zeros((num_seqs,  pts_dataset, self.num_clusters))
            MU = np.zeros((num_seqs, self.num_clusters, 2))
            RADI = np.zeros((num_seqs, pts_dataset, 1))
            ANGLE = np.zeros((num_seqs, pts_dataset, 1))
            for n in range(num_seqs):
                ob, state, mu, radi, angle = self.sim_mixture_triangles(num_seqs)
                OB[n] = ob
                STATE[n] = state
                MU[n] = mu
                RADI[n] = radi
                ANGLE[n] = angle
        else:
            print('ERROR : shape name undefined!')
            exit(0)

        np.save(path + '/ob', OB)
        np.save(path + '/state', STATE)
        np.save(path + '/mu', MU)
        np.save(path + '/angle', ANGLE)
