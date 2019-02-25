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

def plot_clusters(Xs, mus_true, covs_true, K):
    Xs = Xs.data.numpy()
    mus_true = mus_true.data.numpy()
    covs_true = covs_true.data.numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    for k in range(K):
        plot_cov_ellipse(cov=covs_true[k], pos=mus_true[k], nstd=2, ax=ax, alpha=0.5)

    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])
    plt.show()
