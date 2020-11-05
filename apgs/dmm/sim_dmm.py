import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

class Sim_Rings():
    """
    simulate instances of DMM
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
        self.N, self.K, self.D = N, K, D
        self.Nk = int(self.N / self.K)
        self.period = period
        self.mu_std = mu_std
        self.noise_std = noise_std
        self.radi = radi

    def sim_one_ring(self):
        pts_edge = int(self.Nk / self.period)
        angles = np.linspace(0, 2 * math.pi, pts_edge, endpoint=True)
        pointwise_interval = angles[1] - angles[0]
        rand_shift = np.random.uniform(low=0.0, high=pointwise_interval, size=self.period)
        angles_list = []
        for i in range(self.period):
            angles_list.append(angles+rand_shift[i])
        angles = np.concatenate(angles_list, 0)
        assert angles.shape == (self.Nk,), "ERROR! angle variable has unexpected shape."
        # N = angles.shape[0]
        radis = np.ones(self.Nk) * self.radi
        noise = np.random.normal(0.0, self.noise_std, (self.Nk, 2))
        x = np.cos(angles) * radis
        y = np.sin(angles) * radis
        pos = np.concatenate((x[:, None], y[:, None]), -1)
        pos = pos + noise
        return pos, radis, angles

    def sim_one_dmm(self):
        ob = []
        state = []
        radi = []
        angle = []
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
            ob, state, mu, _, _ = self.sim_one_dmm()
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
        for s in range(num_seqs):
            ob, _, _, _, _ = self.sim_one_dmm()
            OB[s] = ob
        print('saving to %s' % os.path.abspath(PATH))
        np.save(PATH + 'ob', OB)
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('DMM_DATA')
    parser.add_argument('--num_instances', default=10000, type=int)
    parser.add_argument('--data_path', default='../../data/dmm/')
    parser.add_argument('--N', default=200, help='number of points in one DMM instance')
    parser.add_argument('--K', default=4, help='number of clusters in one DMM instance')
    parser.add_argument('--D', default=2, help='dimension of data points')
    parser.add_argument('--period', default=2, help='number of circles in each ring')
    parser.add_argument('--mu_std', default=3.0, help='standard deviation of the centers of rings')
    parser.add_argument('--noise_std', default=0.1, help='standard deviation of Gaussian noise')
    parser.add_argument('--radius', default=2.0, help='raidus of the ring')
    args = parser.parse_args()
    simulator = Sim_Rings(args.N, args.K, args.D, args.period, args.mu_std, args.noise_std, args.radius)
    simulator.sim_save_data(args.num_instances, args.data_path)