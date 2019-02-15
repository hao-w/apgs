## By Eli
import torch as to
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import numpy as np

class BallSim():
    def __init__(self, timesteps=10000, dt=0.1, box_lenght=100, obs_cov=None, obs_std=0.1):
        self.ball_rad = 3
        self.box_length = box_lenght

        self.d = 2
        self.timesteps = timesteps
        self.dt = dt
        if obs_cov is None:
            self.obs_cov = obs_std*to.eye(self.d)
            

        self.X0 = to.zeros(3*self.d)
        self.X0[2:4] = to.randn(2)

        self.X = to.zeros((timesteps, 3*self.d))
        self.Y = to.zeros((timesteps, self.d))

    def simulate(self):
        self.X[0] = self.X0
        self.Y[0] = MultivariateNormal(self.X[0, 0:2], self.obs_cov).sample()

        for t in range(0, self.timesteps-1):
            self.X[t+1, 2:4] = to.where(to.abs(self.X[t, 0:2]) > self.box_length/2 - self.ball_rad, -self.X[t, 2:4], self.X[t, 2:4])
            self.X[t+1, 0:2] = self.X[t, 0:2] + self.dt * self.X[t+1, 2:4]
            self.Y[t+1] = to.distributions.multivariate_normal.MultivariateNormal(self.X[t+1, 0:2], self.obs_cov).sample()

    def generate_frames(self, path, img_size=28, frames=100, filename='ball_sim'):
        frame_offset = self.timesteps // frames
        self.simulate()
        for t in range(0, self.timesteps-1):
            if t%frame_offset == 0:
                self.generate_frame(t, path, filename, img_size)

    def generate_frame(self, t, path, filename, img_size):
        full_path = path+'/'+filename+'_'+str(t)+'.png'
        Y = self.Y.numpy()[t]
        dpi = 100.
        
        fig = plt.figure(frameon=False)
        fig.set_size_inches(img_size/dpi,img_size/dpi)
        ax = plt.Axes(fig, [0 , 0 , 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.set_xlim((-self.box_length/2, self.box_length/2))
        ax.set_ylim((-self.box_length/2, self.box_length/2))
        
        ball = plt.Circle((Y[0], Y[1]), self.ball_rad, color='b')
        ax.add_artist(ball)
        
        plt.savefig('full_path', dpi=dpi)
        plt.close()

if __name__ == '__main__':
    sim = BallSim(timesteps=100, dt=.1)
    sim.generate_frames("./ball_imgs/nolabel", img_size=28, frames=1)