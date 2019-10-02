import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import gzip
import math
import numpy as np
import os
import torch
from torch.distributions.uniform import Uniform
from torch.nn.functional import affine_grid, grid_sample
class BouncingMNIST():
    def __init__(self, mnist_path, path, timesteps, num_digits, step_length, file_size):
        '''
        X : coordinates
        V : velocity
        '''
        super(BouncingMNIST, self).__init__()
        self.mnist_path = mnist_path
        self.path = path
        self.timesteps = timesteps
        self.num_digits = num_digits
        self.image_size = 64
        self.digit_size = 28
        self.step_length = step_length
        self.file_size = file_size

    def load_mnist(self):
      # Load MNIST dataset for generating training data.
        path = os.path.join(self.mnist_path, 'train-images-idx3-ubyte.gz')
        with gzip.open(path, 'rb') as f:
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
            X_new = X[t] + V[t] * self.step_length
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

    def sim_tjs(self, num_tjs):
        Xs = []
        Vs = []
        a2 = 0.5**2
        while(True):
            x0 = Uniform(-1, 1).sample((num_tjs, 2))
            if ((x0[0] - x0[1])**2).sum() > a2:
                break
        for i in range(num_tjs):
            x, v = self.sim_trajectory(init_xs=x0[i])
            Xs.append(x.unsqueeze(0))
            Vs.append(v.unsqueeze(0))
        return torch.cat(Xs, 0), torch.cat(Vs, 0)

    def sim_bouncing_mnist(self, mnist):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        s_factor = self.image_size / self.digit_size
        t_factor = (self.image_size - self.digit_size) / self.digit_size
        Video = []
        TJ = []
        inds = torch.randint(0, mnist.shape[0], (self.num_digits,))
        Xs, Vs = self.sim_tjs(num_tjs=self.num_digits)
        for n in range(self.num_digits):
            digit_image = torch.from_numpy(mnist[inds[n]] / 255.0).float()
            S = torch.Tensor([[s_factor, 0], [0, s_factor]]).repeat(self.timesteps, 1, 1)
            Thetas = torch.cat((S, Xs[n].unsqueeze(-1) * t_factor), -1)
            grid = affine_grid(Thetas, torch.Size((self.timesteps, 1, self.image_size, self.image_size)))
            Video.append(grid_sample(digit_image.repeat(self.timesteps, 1, 1).unsqueeze(1), grid, mode='nearest'))
            TJ.append(Xs[n].unsqueeze(0))
            # Init_V.append(V[0.unsqueeze()])
        Video = torch.cat(Video, 1).sum(1).clamp(min=0.0, max=1.0)
        return Video, torch.cat(TJ, 0)

    def sim_videoes(self, num_videoes):
        Videoes = []
        TJs = []
        mnist = self.load_mnist()
        for i in range(num_videoes):
            video, tj = self.sim_bouncing_mnist(mnist)
            Videoes.append(video.unsqueeze(0))
            TJs.append(tj.unsqueeze(0))
        return torch.cat(Videoes, 0) , torch.cat(TJs, 0)## NUM_VIDEOES * K * N * 2

    def save_data(self, num_videoes):

        num_files = int(num_videoes / self.file_size)
        for i in range(num_files):
            data, TJs = self.sim_videoes( self.file_size)
            np.save(self.mnist_path + '/bmnist/ob_%d' % i, data)
            np.save(self.mnist_path + '/bmnist/tj_%d' % i, TJs)

    def viz_moving_mnist(self, fs, num_videoes):
        data, _ = self.sim_videoes(num_videoes)
        num_cols = self.timesteps
        num_rows = num_videoes
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.05, hspace=0.05)
        fig = plt.figure(figsize=(fs, fs * num_rows / num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                ax = fig.add_subplot(gs[i, j])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(data[i, j, :, :], cmap='gray', vmin=0.0, vmax=1.0)
