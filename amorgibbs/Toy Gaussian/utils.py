import numpy as np
import torch
from objectives import *

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

# def SNR(grads, N, beta1, beta2):
#     ltr1, ltr2 = init_tril(N, beta1, beta2)
#     grads2 = grads ** 2
#     E_g = (ltr1 * grads).sum(-1) * (1 - beta1) / (1 - ltr1[:, 0] * beta1)
#     E_g2 = (ltr2 * grads2).sum(-1) * (1 - beta2) / (1 - ltr2[:, 0] * beta2)
#     var = E_g2 - E_g ** 2
#     snr = E_g2 / (var + 1e-8)
#     return E_g, E_g2, var, snr

# def SNR(obj, q_mu, q_sigma, p_mu, p_sigma, num_samples, optimizer, num_samples_snr, alpha):
#     Grad_mu = []
#     Grad_sigma = []
#     for i in range(num_samples_snr):
#         optimizer.zero_grad()
#         loss, _, _, _, _ = obj(q_mu, q_sigma, p_mu, p_sigma, num_samples, alpha)
#         loss.backward()
#         Grad_mu.append(- q_mu.grad.item())
#         Grad_sigma.append(- q_sigma.grad.item())
#
#     snr_mu, var_mu = stats(np.array(Grad_mu))
#     snr_sigma, var_sigma = stats(np.array(Grad_sigma))
#     optimizer.zero_grad()
#     return (snr_mu + snr_sigma) / 2, (var_mu + var_sigma) / 2



def train(obj, q_mu, q_sigma, p_mu, p_sigma, steps, num_samples, optimizer, filename, num_batches):
    LOSSs = []
    EUBOs = []
    ELBOs = []
    IWELBOs = []
    ESSs = []
    SNRs = []
    VARs = []
    KLs = []
    flog = open('results/log-' + filename + '.txt', 'w+')
    flog.write('EUBO, ELBO, IWELBO, ESS, SNR, VAR\n')
    flog.close()
    time_start = time.time()
    for i in range(steps):
        # optimizer.zero_grad()
        # snr, var = SNR(obj, q_mu, q_sigma, p_mu, p_sigma, num_samples, num_batches, optimizer, alpha)
        # SNRs.append(snr.item())
        # VARs.append(var.item())

        optimizer.zero_grad()
        loss, eubo, elbo, iwelbo, ess = obj(q_mu, q_sigma, p_mu, p_sigma, num_samples, num_batches=None)
        loss.backward()
        optimizer.step()
        kl_ex = kl_normal_normal(q_mu, q_sigma, p_mu, p_sigma).mean()
        LOSSs.append(loss.item())
        EUBOs.append(eubo.item())
        IWELBOs.append(iwelbo.item())
        ELBOs.append(elbo.item())
        ESSs.append(ess.item())
        KLs.append(kl_ex.item())

        # flog = open('results/log-' + filename + '.txt', 'a+')
        # flog.write(str(eubo.item()) + ', ' + str(elbo.item()) + ', ' + str(iwelbo.item()) + ', ' + str(ess.item()) + ', ' +
        #            str(snr.item()) + ', ' + str(var.item()) + '\n')

        # if i % 1000 == 0:
            # time_end = time.time()
            # print('iteration:%d, EUBO:%.3f, ELBO:%.3f, IWELBO:%.3f, ESS:%.3f, KL:%.3f (%ds)' % (i, eubo, elbo, iwelbo, ess, kl_ex, (time_end - time_start)))
            # time_start = time.time()
    return LOSSs, EUBOs, ELBOs, IWELBOs, ESSs, SNRs, VARs, KLs

def stats(grads):
    E_g2 = (grads ** 2).mean()
    E_g = grads.mean()
    E2_g = E_g * E_g
    Var_g = E_g2 - E2_g
    SNR_g = E_g2 / Var_g
    return SNR_g, Var_g

def kl_normal_normal(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow(2)
    t1 = ((p_mean - q_mean) / q_std).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
