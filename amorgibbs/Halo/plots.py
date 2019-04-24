import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.gridspec as gridspec   


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip
    
def plot_rings(obs, states, K, bound):
    colors = ['r', 'b', 'gold']
    assignments = states.argmax(-1)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    for k in range(K):
        obs_k = obs[np.where(assignments==k)]
        ax.scatter(obs_k[:,0], obs_k[:, 1], c=colors[k])
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])
