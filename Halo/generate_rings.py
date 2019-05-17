import matplotlib.pyplot as plt
import torch
import numpy as np
import math

def ring(N, period, rad_low, rad_high, noise_std):
    rad = np.random.uniform(rad_low, rad_high, 1)
    rads = np.ones(N) * rad
    noise = np.random.normal(0.0, noise_std, N)
    rads = rads + noise
    angles = np.linspace(0, period * 2 * math.pi, N, endpoint=False)
    x = np.cos(angles) * rads
    y = np.sin(angles) * rads
    pos = np.concatenate((x[:, None], y[:, None]), -1)
    return pos, rad

def rings(K, bound, N, rad_low, rad_high, period, noise_std):
    obs = []
    states = []
    rads = []
    mus = np.random.uniform(-bound+rad_high, bound-rad_high, (K, 2))
    for k in range(K):
        pos, rad = ring(N, period, rad_low, rad_high, noise_std)
        state = np.zeros(K)
        state[k] = 1
        pos = pos + mus[k]
        obs.append(pos)
        states.append(np.tile(state, (N, 1)))
        rads.append(rad[None, :])
    return np.concatenate(obs, 0), np.concatenate(states, 0), mus, np.concatenate(rads, 0)


def ring_fixed_rad(N, period, radius, noise_std):
    rads = np.ones(N) * radius
    noise = np.random.normal(0.0, noise_std, N)
    rads = rads + noise
    angles = np.linspace(0, period * 2 * math.pi, N, endpoint=False)
    x = np.cos(angles) * rads
    y = np.sin(angles) * rads
    pos = np.concatenate((x[:, None], y[:, None]), -1)
    return pos

def rings_fixed_rad(K, bound, N, radius, period, noise_std):
    obs = []
    states = []

    mus = np.random.uniform(-bound + radius, bound-radius, (K, 2))
    for k in range(K):
        pos = ring_fixed_rad(N, period, radius, noise_std)
        state = np.zeros(K)
        state[k] = 1
        pos = pos + mus[k]
        obs.append(pos)
        states.append(np.tile(state, (N, 1)))
    return np.concatenate(obs, 0), np.concatenate(states, 0), mus
