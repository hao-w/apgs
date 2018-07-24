import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

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

def plot_clusters(Xs, mus, covs):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.plot(Xs[:,0], Xs[:,1], 'ro')
    plot_cov_ellipse(cov=covs[0], pos=mus[0], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs[1], pos=mus[1], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=covs[2], pos=mus[2], nstd=2, ax=ax, alpha=0.5)
    plt.show()

def plot_velocity_circle(v):
    fig = plt.figure(figsize=(4,4))
    ax = fig.gca()
    ax.scatter(v[:,0], v[:,1], label='z=1')
    ax.scatter(-v[:,0], v[:,1], label='z=2')
    ax.scatter(v[:,0], -v[:,1], label='z=3')
    ax.scatter(-v[:,0], -v[:,1], label='z=4')
    ax.set_xlabel('x velocity')
    ax.set_ylabel('y velocity')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)

def plot_transition(As_pred, As_true, vmax):
    As_infer = As_pred / As_pred.sum(-1)[:, :, None]
    As_infer = As_infer.mean(0)
    fig3 = plt.figure(figsize=(8,8))
    ax1 = fig3.add_subplot(1, 2, 1)
    infer_plot = ax1.imshow(As_infer, cmap='Greys', vmin=0, vmax=vmax)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('inferred averaged transition matrix')
    ax2 = fig3.add_subplot(1, 2, 2)
    As_true_ave = As_true.mean(0)
    true_plot = ax2.imshow(As_true_ave, cmap='Greys', vmin=0, vmax=vmax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('true averaged transition matrix')
    cbaxes = fig3.add_axes([0.95, 0.32, 0.02, 0.36])
    cb = plt.colorbar(true_plot, cax = cbaxes)
    return As_infer, As_true_ave
