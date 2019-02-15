import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from torch.distributions.uniform import Uniform
import time

def generate_seqs(num_seqs, T, K, D, bound, noise_ratio,  dt=1.0, radius=0.0):
    ## generate multiple trajectories
    Mus = torch.zeros((num_seqs, K, D))
    Covs = torch.zeros((num_seqs, K, D, D))
    Disps = torch.zeros((num_seqs, T, D))
    As = torch.zeros((num_seqs, K, K))
    Zs = torch.zeros((num_seqs, T, K))
    Pis = torch.zeros((num_seqs, K))
    for s in range(num_seqs):
        disp, direction_state, mus, covs, A, pi, position = generate_seq(T, K, bound, noise_ratio,  dt=1.0, radius=0.0)
        Disps[s] = disp
        As[s] = A
        Zs[s] = direction_state
        Mus[s] = mus
        Covs[s] = covs
        Pis[s] = pi
    return Disps, Zs, As, Mus, Covs, Pis

def generate_seq(T, K, bound, noise_ratio,  dt=1.0, radius=0.0):
    noise_cov = torch.eye(2) * noise_ratio
    A = torch.zeros((K, K))
    STATE = torch.zeros((T+1, K)) # since I want to make the displacement of length T, I need the coordinates of length T+1
    init_state = intialization(dt, bound)
    box_bound = torch.FloatTensor([-1, 1, -1, 1]) * (bound - radius*2) ## radius is 0 by default, but need to be set to positve when generate images dataset, which is not used in the HMM model
    Z = torch.zeros((T+1, K))
    STATE[0] = init_state
    for i in range(1, T+1):
        prev_state = STATE[i-1]
        state = step(prev_state, box_bound, noise_cov)
        STATE[i] = state
        new_state = compute_state(state[2:])
        Z[i, new_state] = 1
        if i != 1:
            A[old_state, new_state] += 1
        old_state = new_state
    disp = (STATE[1:] - STATE[:-1])[:, :2]
    A[A == 0.0] = 1e-6
    A_sum = A.sum(1)
    A = A / A_sum[:, None]
    direction_state = Z[1:]
    covs = noise_cov.unsqueeze(0).repeat(K, 1, 1)
    dirs = torch.FloatTensor([[1, 1], [1, -1], [-1, -1], [-1, 1]])
    mus = torch.abs(init_state[2:]).unsqueeze(0).repeat(K, 1) * dirs
    pi = torch.ones(K) * (1. / K)
    return disp, direction_state, mus, covs, A, pi, STATE[:, :2]


def intialization(dt, bound):
    init_state = torch.zeros(4)
    abs_init_position = Uniform(torch.FloatTensor([0.0, 0.0]), torch.FloatTensor([bound, bound])).sample()
    random_direction = torch.from_numpy(np.random.choice([-1.0, 1.0], 2)).float()
    init_state[:2] = abs_init_position * random_direction
    init_v = torch.rand(2) * torch.from_numpy(np.random.choice([-1.0, 1.0], 2)).float()
    v_norm = ((init_v ** 2 ).sum()) ** 0.5 ## compute norm for each initial velocity
    init_v = init_v / v_norm * dt ## to make the velocity lying on a circle
    init_state[2:] = init_v
    return init_state


def step(state, box_bound, noise_cov):
    xy_new = state[ :2] + mvn(state[2: ], noise_cov).sample()
    while(True):
        if xy_new[0] < box_bound[0]:
            x_over = abs(box_bound[0] - xy_new[0])
            xy_new[0] = box_bound[0] + x_over
            state[2] *= -1
        elif xy_new[0] > box_bound[1]:
            x_over = abs(box_bound[1] - xy_new[0])
            xy_new[0] = box_bound[1] - x_over
            state[2] *= -1
        elif xy_new[1] < box_bound[2]:
            y_over = abs(box_bound[2] - xy_new[1])
            xy_new[1] = box_bound[2] + y_over
            state[3] *= -1
        elif xy_new[1] > box_bound[3]:
            y_over = abs(box_bound[3] - xy_new[1])
            xy_new[1] = box_bound[3] - y_over
            state[3] *= -1
        else:
            state[:2] = xy_new
            break
    return state

def compute_state(velocity):
    if velocity[0] > 0 and velocity[1] > 0:
        return 0
    if velocity[0] > 0 and velocity[1] <= 0:
        return 1
    if velocity[0] <= 0 and velocity[1] <= 0:
        return 2
    if velocity[0] <= 0 and velocity[1] > 0:
        return 3

# def generate_frames(STATE, Boundary, pixels, dpi, radius, seq_ind):
#     length_states = STATE.shape[0]
#     for s in range(length_states):
#         fig = plt.figure(frameon=False)
#         fig.set_size_inches(pixels/dpi, pixels/dpi)
#         ax = plt.Axes(fig, [0 , 0 , 1, 1])
#         ax.set_axis_off()
#         fig.add_axes(ax)
#         ball = plt.Circle((STATE[s, 0], STATE[s, 1]), radius, color='black')
#         ax.set_xlim((-Boundary, Boundary))
#         ax.set_ylim((-Boundary, Boundary))
#         ax.add_artist(ball)
#         plt.savefig('images/%s-%s.png' % (str(seq_ind), str(s)),dpi=dpi)
#         plt.close()
#
# def generate_pixels(T, K, num_seq, Boundary, dt, noise_ratio, pixels, dpi, radius):
#     init_v = init_velocity(dt)
#     for s in range(num_seq):
#         STATE, mu_ks, cov_ks, Pi, Y, A_true, Zs_true = generate_seq(T, K, dt, Boundary, noise_ratio, radius)
#         generate_frames(STATE, Boundary, pixels, dpi, radius, s)
#     print('dataset generate completed!')
