import matplotlib.pyplot as plt
import numpy as np

def plot_results(dEUBOs, dELBOs, dIWELBOs, dESSs, dSNRs, dVARs, num_samples, num_samples_snr, steps, lr, ests, fs=10):
    fig = plt.figure(figsize=(fs,fs))
    colors = {'mc':'green', 'iwae': 'red', 'iwae-dreg': 'blue',
              'rws': 'deepskyblue', 'rws-dreg': 'firebrick', 'stl': 'black'}
    fig = plt.figure(figsize=(fs,fs))
    ax = fig.subplots(5, 3, gridspec_kw={'wspace':0.1, 'hspace':0.1})

    for i, est in enumerate(ests):
        EUBOs = dEUBOs[est]
        ELBOs = dELBOs[est]
        IWELBOs = dIWELBOs[est]
        ESSs = dESSs[est]
        SNRs = dSNRs[est]
        VARs = dVARs[est]
        if est == 'mc':
            ax[0, 0].plot(EUBOs, c=colors[est], label=est)
            ax[1, 0].plot(ELBOs, c=colors[est], label=est)
            ax[2, 0].plot(IWELBOs, c=colors[est], label=est)
            ax[3, 0].plot(SNRs, c=colors[est], label=est + '-snr')
            ax[3, 0].plot(VARs, marker='o', c=colors[est], label=est + '-var')
            ax[4, 0].plot(ESSs, c=colors[est], label=est)
        elif est == 'iwae' or est == 'iwae-dreg':
            ax[0, 1].plot(EUBOs, c=colors[est], label=est)
            ax[1, 1].plot(ELBOs, c=colors[est], label=est)
            ax[2, 1].plot(IWELBOs, c=colors[est], label=est)
            ax[3, 1].plot(SNRs, c=colors[est], label=est + '-snr')
            ax[3, 1].plot(VARs, marker='o', c=colors[est], label=est + '-var')
            ax[4, 1].plot(ESSs, c=colors[est], label=est)
        elif est =='rws' or est == 'rws-dreg':
            ax[0, 2].plot(EUBOs, c=colors[est], label=est)
            ax[1, 2].plot(ELBOs, c=colors[est], label=est)
            ax[2, 2].plot(IWELBOs, c=colors[est], label=est)
            ax[3, 2].plot(SNRs, c=colors[est], label=est + '-snr')
            ax[3, 2].plot(VARs, marker='o', c=colors[est], label=est + '-var')
            ax[4, 2].plot(ESSs, c=colors[est], label=est)

    ax[0, 1].set_title('EUBO')
    ax[1, 1].set_title('ELBO')
    ax[2, 1].set_title('IWELBO')
    ax[3, 1].set_title('SNR and Variance')
    ax[4, 1].set_title('ESS')

    for i in range(5):
        for j in range(3):
            ax[i, j].legend(fontsize=14)
            ax[i, j].tick_params(labelsize=14)
            if i == 3:
                ax[i, j].set_yscale('log')
    plt.savefig('results/results-%dsamples-%dSNRsamples-%dsteps-%flr.svg' % (num_samples, num_samples_snr, steps, lr))

def plot_results_simplified(dLOSSs, dESSs, dKLs, num_samples, num_batches, steps, lr, ests, fs=30):
    fig = plt.figure(figsize=(fs,fs))
    colors = {'mc':'green', 'iwae': 'red', 'iwae-dreg': 'blue',
              'rws': 'deepskyblue', 'rws-dreg': 'firebrick', 'stl': 'black'}
    fig = plt.figure(figsize=(fs,fs))
    ax = fig.subplots(3, 1, gridspec_kw={'wspace':0.1, 'hspace':0.1})

    for i, est in enumerate(ests):
        LOSSs = dLOSSs[est]
        ESSs = dESSs[est]
        KLs = dKLs[est]
        if est == 'mc' or est == 'iwae' or est == 'iwae-dreg':
            ax[0].plot(- LOSSs, c=colors[est], label= 'ELBO' + est)
            ax[1].plot(KLs, c=colors[est], label=est)
            ax[2].plot(np.array(ESSs), c=colors[est], label=est)
        elif est =='rws' or est == 'rws-dreg':
            ax[0].plot(LOSSs, c=colors[est], label= 'EUBO' + est)
            ax[1].plot(KLs, c=colors[est], label=est)
            ax[2].plot(np.array(ESSs), c=colors[est], label=est)
    ax[0].set_title('Objectives')
    ax[1].set_title('exclusive KL')
    ax[2].set_title('ESS')

    for i in range(3):
        ax[i].legend(fontsize=12)
        ax[i].tick_params(labelsize=12)
        if i == 1:
            ax[i].set_yscale('log')
    plt.savefig('results/results-%dsamples-%dSNRsamples-%dsteps-%.4flr.svg' % (num_samples, num_batches, steps, lr))
