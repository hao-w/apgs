import torch as to
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

class ball_sim():
    def __init__(self, T=10000, dt=0.1, frames=100, name='ball_sim', box_lenght=100, obs_cov=None, obs_std=0.1):
        self.name = name
        self.ball_rad = 3.

        self.d = 2
        self.T = T
        self.dt = dt
        if obs_cov is None:
            self.obs_cov = obs_std*to.eye(self.d)
            ??

        self.frames = frames
        self.box_length = box_lenght?
        self.X0 = to.zeros(3*self.d)
        self.X0[2:4] = to.randn(2)

        self.X = to.zeros((T, 3*self.d))
        self.Y = to.zeros((T, self.d))

    def simulate(self):
        self.X[0] = self.X0
        self.Y[0] = MultivariateNormal(self.X[0, 0:2], self.obs_cov).sample()

        for t in range(0, self.T-1):
            self.X[t+1, 2:4] = to.where(to.abs(self.X[t, 0:2]) > self.box_length/2 - self.ball_rad, -self.X[t, 2:4], self.X[t, 2:4])
            self.X[t+1, 0:2] = self.X[t, 0:2] + self.dt * self.X[t+1, 2:4]
            self.Y[t+1] = to.distributions.multivariate_normal.MultivariateNormal(self.X[t+1, 0:2], self.obs_cov).sample()

    def generate_frames(self, path):
        self.simulate()
        for t in range(0, self.T-1):
            if t%self.frames == 0:
                self.generate_frame(t, path)
                print(self.X[t])
                
    def generate_frame(self, t, path):
        Y = self.Y.numpy()[t]
        ax = plt.gcf().gca()
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim((-self.box_length/2, self.box_length/2))
        ax.set_ylim((-self.box_length/2, self.box_length/2))
        ball = plt.Circle((Y[0], Y[1]), self.ball_rad, color='b')
        ax.add_artist(ball)
        plt.savefig(path+'/'+self.name+'_'+str(t//self.frames)+'.png')

if __name__ == '__main__':
    sim = ball_sim(T=10000, dt=.1, frames=100)
    sim.generate_frames("./bball_imgs")