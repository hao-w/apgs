import matplotlib.pyplot as plt
import numpy as np

def plot_results(EUBO, ELBO, ESS, num_samples, snr_mu, snr_sigma):
    fig = plt.figure(figsize=(10,10))
    ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
    plt.tight_layout()
    ax1.plot(EUBO, 'r', label='EUBOs')
    ax1.plot(ELBO, 'b', label='ELBOs')
    ax1.legend()
    ## SNR
    ax2.plot(snr_sigma, label='SNR_sigma')
    ax2.plot(snr_mu, label='SNR_mu')
    ax2.legend()
    ax2.set_ylim([-1,5])
    ## ESS
    ess_ratio = np.array(ESS) / num_samples
    ave_ess = np.reshape(ess_ratio, (-1, 10)).mean(-1)
    N = ave_ess.shape[0]
    ax3.plot(np.arange(N) * 10, ave_ess, 'go', label='ESS')
    ax3.set_ylim([0, 1])
    