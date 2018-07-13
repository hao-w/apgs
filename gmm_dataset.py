import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, multinomial
from matplotlib.patches import Ellipse

K = 3
T = 500
A = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
mus = np.array([[1,1], [2,10], [10, 5.5]])
cov1 = np.expand_dims(np.array([[3, 0],[0, 0.5]]), 0)
cov2 = np.expand_dims(np.array([[1, 0.7],[0.7, 1]]), 0)
cov3 = np.expand_dims(np.array([[1, -0.8],[-0.8, 1]]), 0)
covs = np.concatenate((cov1, cov2, cov3), axis=0)
Pi = np.array([1/3, 1/3, 1/3])

def sample_state(P):
    s = np.nonzero(multinomial.rvs(1, P, size=1, random_state=None)[0])[0][0]
    return s

def sampling():
    Xs = []
    for t in range(T):
        if t == 0:
            zt = sample_state(Pi)
            xt = multivariate_normal.rvs(mus[zt], covs[zt])
            Xs.append(xt)
            ztp1 = sample_state(A[zt])
        else:
            zt = ztp1
            xt = multivariate_normal.rvs(mus[zt], covs[zt])
            Xs.append(xt)
            ztp1 = sample_state(A[zt])
    Xs = np.asarray(Xs)
    return Xs

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

def plot_samples(Xs):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    plot_cov_ellipse(cov=covs[0], pos=mus[0], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs[1], pos=mus[1], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs[2], pos=mus[2], nstd=2, ax=ax, alpha=0.5)
    plt.show()
