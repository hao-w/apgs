import numpy as np

def init_tril(T, beta1, beta2):
    lbeta1 = np.log(beta1)
    lbeta2 = np.log(beta2)
    ltr = np.eye(T)
    tt = np.arange(T)
    for n in range(T):
        ltr[n:, n] = tt[:T-n]
    tril1 = np.tril(np.exp(np.ones((T, T)) * lbeta1 * ltr))
    tril2 = np.tril(np.exp(np.ones((T, T)) * lbeta2 * ltr))
    return tril1, tril2

def SNR(grads, N, beta1, beta2):
    ltr1, ltr2 = init_tril(N, beta1, beta2)
    grads2 = grads ** 2
    E_g = (ltr1 * grads).sum(-1) * (1 - beta1) / (1 - ltr1[:, 0] * beta1)
    E_g2 = (ltr2 * grads2).sum(-1) * (1 - beta2) / (1 - ltr2[:, 0] * beta2)
    var = E_g2 - E_g ** 2
    snr = E_g**2 / var
    return E_g, E_g2, var, snr