import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch
from probtorch.util import log_sum_exp
from bokeh.plotting import figure, output_notebook, show


def shuffler(train_dataset):
    index = np.arange(train_dataset.shape[0])
    np.random.shuffle(index)
    return train_dataset[index, :]

def NUM_ITERS(train_dataset, batch_size):
    remainder = train_dataset.shape[0] % batch_size
    if remainder == 0:
        num_iters = int(train_dataset.shape[0] / batch_size)
    else:
        num_iters = int(((train_dataset.shape[0] - remainder) / batch_size) + 1)
    return num_iters

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

def plot_clusters(Y, mus, covs, K):
    cmap = plt.cm.get_cmap('hsv', K)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(Y[:,0], Y[:,1], 'k.', markersize=4.0)
    for i in range(K):
        plot_cov_ellipse(cov=covs[i],
                         pos=mus[i],
                         nstd=2,
                         ax=ax,
                         color=cmap(i),
                         alpha=0.1)

    plt.show()

def log_expectation_wi_single(nu, W, D):
    ds = (nu + 1 - (torch.arange(D).float() + 1)) / 2.0
    return  - D * torch.log(torch.Tensor([2])) + torch.log(torch.det(W)).float() - torch.digamma(ds).sum()

def log_expectation_wi(nu_ks, W_ks, D, K):
    log_expectations = torch.zeros(K)
    for k in range(K):
        log_expectations[k] = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
    return log_expectations

def log_expectation_dir(alpha_hat, K):
    log_expectations = torch.zeros(K)
    sum_digamma = torch.digamma(alpha_hat.sum())
    for k in range(K):
        log_expectations[k] = torch.digamma(alpha_hat[k]) - sum_digamma
    return log_expectations


def log_C(alpha):
    return torch.lgamma(alpha.sum()) - (torch.lgamma(alpha)).sum()
