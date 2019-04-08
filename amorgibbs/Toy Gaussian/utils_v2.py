import numpy as np
import torch
from objectives_v2 import *

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

def train(obj, q_mu, q_sigma, p_mu, p_sigma, steps, num_samples, optimizer):
    LOSSs = []
    ESSs = []
    KLs = []
    for i in range(steps):
        optimizer.zero_grad()
        loss, ess = obj(q_mu, q_sigma, p_mu, p_sigma, num_samples)
        loss.backward()
        optimizer.step()
        LOSSs.append(loss.item())

        kl_ex = kl_normal_normal(q_mu, q_sigma, p_mu, p_sigma).sum()
        ESSs.append(ess.item())
        KLs.append(kl_ex.item())

    return LOSSs, ESSs, KLs

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
