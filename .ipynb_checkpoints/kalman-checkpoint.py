import autograd.numpy as np
from autograd.numpy.linalg import inv, det
from functools import reduce
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gamma, invgamma
from scipy.special import digamma, polygamma
from scipy.special import gamma as gafun
from util import eigen

##### Forward #####
def forward_step(mu_last, sigma_last, yt, K, *natstats_list):
    # unzip the list of natstats
    E_A, E_ATA, E_R_inv, E_C, E_R_inv_C, E_CT_R_inv_C = natstats_list
    # mu and sigma for xt
    sigma_star_last = inv(inv(sigma_last) + E_ATA.T)
    sigma_xt = inv(np.eye(K) + E_CT_R_inv_C - eigen(E_A.T, sigma_star_last))
    mu_xt = np.dot(sigma_xt, \
                   np.dot(E_R_inv_C.T, yt) \
                   + reduce(np.dot, [E_A, sigma_star_last, inv(sigma_last), mu_last]))
    # predictive distribution of yt
    # yt_sigma = inv(E_R_inv - eigen(E_R_inv_C.T, sigma_xt))
    # yt_mu = np.dot(yt_sigma, \
    #               reduce(np.dot, [E_R_inv_C, sigma_xt, E_A, sigma_star_last, inv(sigma_last), mu_last]))
    # log of yt
    t1 = np.log(2*np.pi)
    t2 = np.log(det(reduce(np.dot, [inv(sigma_last), sigma_star_last, sigma_xt])))
    t3 = eigen(mu_last, inv(sigma_last))
    t4 = eigen(mu_xt, inv(sigma_xt))
    t5 = eigen(yt, E_R_inv)
    t6 = eigen(np.dot(inv(sigma_last), mu_last), sigma_star_last)

    log_yt = -(1 / 2) * (t1 - t2 + t3 - t4 + t5 - t6)

    return mu_xt, sigma_xt, sigma_star_last, log_yt


def forward(mu_x0, sigma_x0, N, T, K, video, *natstats_list):
    log_Z = []
    Sigma_star = np.zeros((T, K, K))
    Mu_xt = np.zeros((T+1, K, 1))
    Sigma_xt = np.zeros((T+1, K, K))
    Mu_xt[0] = mu_x0
    Sigma_xt[0] = sigma_x0
    for t in range(T):
        yt = video[:, t]
        yt.shape = (N, 1)
        if t == 0:
            mu_last = mu_x0
            sigma_last = sigma_x0
        else:
            mu_last = mu_xt
            sigma_last = sigma_xt

        mu_xt, sigma_xt, sigma_star_last, log_yt = forward_step(mu_last, sigma_last, yt, K, *natstats_list)

        log_Z.append(log_yt[0][0])
        Sigma_star[t] = sigma_star_last
        Mu_xt[t+1] = mu_xt
        Sigma_xt[t+1] = sigma_xt
    log_Z = np.array(log_Z).sum()
    return Mu_xt, Sigma_xt, Sigma_star, log_Z


##### Backward #####
def backward_step(psi_t_inv, eta_t, yt, K, *natstats_list):
    E_A, E_ATA, E_R_inv, E_C, E_R_inv_C, E_CT_R_inv_C = natstats_list
    psi_star_t = inv(np.eye(K) + E_CT_R_inv_C + psi_t_inv)
    psi_next = inv(E_ATA - eigen(E_A, psi_star_t))
    term0 = np.dot(E_R_inv_C.T, yt) + np.dot(psi_t_inv, eta_t)
    eta_next = reduce(np.dot, [psi_next, E_A.T, psi_star_t, term0])
    return psi_next, eta_next

def backward(psi_T_inv, eta_T, video, N, T, K, *natstats_list):
    Psi = np.zeros((T+1, K, K))
    Eta = np.zeros((T+1, K, 1))
    Psi[T] = psi_T_inv
    Eta[T] = eta_T

    for t in range(T, 0, -1):
        yt = video[:, t-1]
        yt.shape = (N, 1)
        if t == T:
            psi_t_inv = psi_T_inv
            eta_t = eta_T
        else:
            psi_t_inv = psi_next_inv
            eta_t = eta_next
        psi_next, eta_next = backward_step(psi_t_inv, eta_t, yt, K, *natstats_list)
        print(psi_next)
        psi_next_inv = inv(psi_next)
        Psi[t-1] = psi_next
        Eta[t-1] = eta_next
    return Psi, Eta
