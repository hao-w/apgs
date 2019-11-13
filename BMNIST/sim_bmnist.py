import os
import gzip
import math
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.distributions.uniform import Uniform
from torch.nn.functional import affine_grid, grid_sample


"""
==========
simulate bouncing mnist using the training dataset in mnist
==========
"""
class Sim_BMNIST():
    def __init__(self, timesteps, num_digits, frame_size, delta_t, chunk_size):
        '''
        X : coordinates
        V : velocity
        '''
        super(Sim_BMNIST, self).__init__()
        self.timesteps = timesteps
        self.num_digits = num_digits
        self.frame_size = frame_size
        self.mnist_size = 28 ## by default
        self.delta_t = delta_t
        self.chunk_size = chunk_size ## datasets are dividied into pieces with this number and saved separately

    def load_mnist(self, MNIST_DIR):
        MNIST_PATH = os.path.join(MNIST_DIR, 'train-images-idx3-ubyte.gz')
        if not os.path.exists(MNIST_PATH):
            print('===Downloading MNIST train dataset from \'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\'===')
            if not os.path.exists(MNIST_DIR):
                os.makedirs(MNIST_DIR)
            r = requests.get('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
            open(MNIST_PATH,'wb').write(r.content)
            print('===Saved to \'%s\'===' % MNIST_PATH)

        with gzip.open(MNIST_PATH, 'rb') as f:
            mnist = np.frombuffer(f.read(), np.uint8, offset=16)
            mnist = mnist.reshape(-1, 28, 28)
        return mnist

    def sim_trajectory(self, init_xs):
        ''' Generate a random sequence of a MNIST digit '''
        v_norm = Uniform(0, 1).sample() * 2 * math.pi
        #v_norm = torch.ones(1) * 2 * math.pi
        v_y = torch.sin(v_norm).item()
        v_x = torch.cos(v_norm).item()
        V0 = torch.Tensor([v_x, v_y])
        X = torch.zeros((self.timesteps, 2))
        V = torch.zeros((self.timesteps, 2))
        X[0] = init_xs
        V[0] = V0
        for t in range(0, self.timesteps -1):
            X_new = X[t] + V[t] * self.delta_t
            V_new = V[t]

            if X_new[0] < -1.0:
                X_new[0] = -1.0 + torch.abs(-1.0 - X_new[0])
                V_new[0] = - V_new[0]
            if X_new[0] > 1.0:
                X_new[0] = 1.0 - torch.abs(X_new[0] - 1.0)
                V_new[0] = - V_new[0]
            if X_new[1] < -1.0:
                X_new[1] = -1.0 + torch.abs(-1.0 - X_new[1])
                V_new[1] = - V_new[1]
            if X_new[1] > 1.0:
                X_new[1] = 1.0 - torch.abs(X_new[1] - 1.0)
                V_new[1] = - V_new[1]
            V[t+1] = V_new
            X[t+1] = X_new
        return X, V

    def sim_trajectories(self, num_tjs):
        Xs = []
        Vs = []
        x0 = Uniform(-1, 1).sample((num_tjs, 2))
        # a2 = 0.5**2
        # while(True):
        #     if ((x0[0] - x0[1])**2).sum() > a2:
        #         break
        #     x0 = Uniform(-1, 1).sample((num_tjs, 2))
        for i in range(num_tjs):
            x, v = self.sim_trajectory(init_xs=x0[i])
            Xs.append(x.unsqueeze(0))
            Vs.append(v.unsqueeze(0))
        return torch.cat(Xs, 0), torch.cat(Vs, 0)

    def sim_one_bmnist(self, mnist, mnist_index):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        s_factor = self.frame_size / self.mnist_size
        t_factor = (self.frame_size - self.mnist_size) / self.mnist_size
        bmnist = []
        Xs, Vs = self.sim_trajectories(num_tjs=self.num_digits)
        for k in range(self.num_digits):
            digit_image = torch.from_numpy(mnist[mnist_index[k]] / 255.0).float()
            S = torch.Tensor([[s_factor, 0], [0, s_factor]]).repeat(self.timesteps, 1, 1)
            Thetas = torch.cat((S, Xs[k].unsqueeze(-1) * t_factor), -1)
            grid = affine_grid(Thetas, torch.Size((self.timesteps, 1, self.frame_size, self.frame_size)))
            bmnist.append(grid_sample(digit_image.repeat(self.timesteps, 1, 1).unsqueeze(1), grid, mode='nearest'))
            # TJ.append(Xs[n].unsqueeze(0))
            # Init_V.append(V[0.unsqueeze()])
        bmnist = torch.cat(bmnist, 1).sum(1).clamp(min=0.0, max=1.0)
        return bmnist

    def sim_save_data(self, num_seqs, MNIST_DIR, PATH):
        """
        ==========
        way it saves data:
        if num_seqs <= N, then one round of indexing is enough
        if num_seqs > N, then more than one round is needed
        ==========
        """
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        mnist = self.load_mnist(MNIST_DIR=MNIST_DIR)
        N = mnist.shape[0]
        assert num_seqs > 0, 'number of sequences must be a positive number'
        assert isinstance(num_seqs, int)
        consecutive = torch.arange(N).repeat(1, self.num_digits).squeeze(0)
        mnist_indices = consecutive[torch.randperm(N*self.num_digits)].view(N, self.num_digits) ## generate random indices
        num_seqs_left = num_seqs
        print('Start to simulate bouncing mnist sequences...')
        counter = 1
        while(num_seqs_left > 0):
            time_start = time.time()
            num_this_round = min(self.chunk_size, num_seqs_left)
            bmnists = []
            if mnist_indices.shape[0] < num_this_round: ## more indices are needed for this round
                new_indices = consecutive[torch.randperm(N*self.num_digits)].view(N, self.num_digits)
                mnist_indices = torch.cat((mnist_indices, new_indices), 0)
            for i in range(num_this_round):
                bmnist = self.sim_one_bmnist(mnist=mnist, mnist_index=mnist_indices[i])
                bmnists.append(bmnist.unsqueeze(0))
            mnist_indices = mnist_indices[num_this_round:]
            bmnists = torch.cat(bmnists, 0)
            assert bmnists.shape == (num_this_round, self.timesteps, self.frame_size, self.frame_size), "ERROR! unexpected chunk shape."
            incremental_PATH = PATH + 'ob-%d' % counter
            np.save(incremental_PATH, bmnists)
            counter += 1
            num_seqs_left = max(num_seqs_left - num_this_round, 0)
            time_end = time.time()
            print('(%ds) Simulated %d sequences, saved to \'%s\', %d sequences left.' % ((time_end - time_start), num_this_round, incremental_PATH, num_seqs_left))

    def viz_data(self, MNIST_DIR, num_seqs=5, fs=15):
        mnist = self.load_mnist(MNIST_DIR=MNIST_DIR)
        N = mnist.shape[0]
        mnist_indices = torch.arange(N).repeat(1, self.num_digits).squeeze(0)
        mnist_indices = mnist_indices[torch.randperm(N*self.num_digits)].view(N, self.num_digits)
        num_cols = self.timesteps
        num_rows = num_seqs
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.05, hspace=0.05)
        fig = plt.figure(figsize=(fs, fs * num_rows / num_cols))
        for i in range(num_rows):
            bmnist = self.sim_one_bmnist(mnist, mnist_indices[i])
            for j in range(num_cols):
                ax = fig.add_subplot(gs[i, j])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(bmnist[j], cmap='gray', vmin=0.0, vmax=1.0)
