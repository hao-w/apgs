import matplotlib.pyplot as plt
import torch
import numpy as np
import math

def ring(N, rad, period, noise_std):
    rads = np.ones(N) * rad
    noise = np.random.normal(0.0, noise_std, N)
    rads = rads + noise
    angles = np.linspace(0, period * 2 * math.pi, N, endpoint=False)
    x = np.cos(angles) * rads
    y = np.sin(angles) * rads
    pos = np.concatenate((x[:, None], y[:, None]), -1)
    return pos

def rings(K, bound, N, rad, period, noise_std):
    obs = []
    states = []
    mus = np.random.uniform(-bound + rad, bound-rad, (K, 2))
    for k in range(K):
        pos = ring(N, rad, period, noise_std)
        state = np.zeros(K)
        state[k] = 1
        pos = pos + mus[k]
        obs.append(pos)
        states.append(np.tile(state, (N, 1)))
    return np.concatenate(obs, 0), np.concatenate(states, 0), mus
