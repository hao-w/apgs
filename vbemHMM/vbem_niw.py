from fb_hmm import *
import math
import torch
import numpy as np
from torch import logsumexp
import time

def init_priors(Y, T, K, D):
    alpha_init_0 = torch.ones(K)
    alpha_trans_0 = torch.ones((K, K))
    ms_0 = torch.FloatTensor([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * (1 / math.sqrt(2))
    nu = 6.0
    beta = 1.0
    betas_0 = beta * torch.ones(K)
    nus_0 = nu * torch.ones(K)
    W =  (nu - D - 1) * torch.mm((Y - Y.mean(0)).transpose(0,1), (Y - Y.mean(0))) / (T)
    Ws_0 = W.repeat(K, 1, 1)
    return alpha_init_0, alpha_trans_0, ms_0, betas_0[0], nus_0[0], Ws_0[0]

def init_posterior(Y, T, K, D):
    alpha_init = torch.ones(K)
    alpha_trans = torch.ones((K, K))
    ms = torch.FloatTensor([[1, 1], [1, -1], [-1, -1], [-1, 1]])
    ## In general m_ks should be initialized randomly
    # m_0 = torch.FloatTensor([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * (1 / math.sqrt(2))
    # cov = torch.from_numpy(np.cov(Y.transpose(0,1).data.numpy())).float()
    # m_ks = mvn(m_0, cov).sample((K,))
    nu = 6.0
    beta = 1.0
    betas = torch.ones(K) * beta
    nus = torch.ones(K) * nu
    W =  (nu - D - 1) * torch.mm((Y - Y.mean(0)).transpose(0,1), (Y - Y.mean(0))) / (T)
    Ws = W.repeat(K, 1, 1)
    return alpha_init, alpha_trans, ms, betas, nus, Ws


def quad(a, B):
    return torch.mm(torch.mm(a.transpose(0, 1), B), a)

def log_expectation_wi_single(nu, W, D):
    ds = (nu + 1 - (torch.arange(D).float() + 1)) / 2.0
    return  - D * torch.log(torch.Tensor([2])) + torch.log(torch.det(W)) - torch.digamma(ds).sum()

def log_expectation_wi_single2(nu, W, D):
    ds = (nu + 1 - (torch.arange(D).float() + 1)) / 2.0
    return  D * torch.log(torch.Tensor([2])) + torch.log(torch.det(W)) + torch.digamma(ds).sum()

def log_expectation_wi2(nu_ks, W_ks, D, K):
    log_expectations = torch.zeros(K)
    for k in range(K):
        log_expectations[k] = log_expectation_wi_single2(nu_ks[k], W_ks[k], D)
    return log_expectations

def log_expectation_wi(nu_ks, W_ks, D, K):
    log_expectations = torch.zeros(K)
    for k in range(K):
        log_expectations[k] = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
    return log_expectations

def quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    quadratic_expectations = torch.zeros((K, N))
    for k in range(K):
        quadratic_expectations[k] = D / beta_ks[k] + nu_ks[k] * torch.mul(torch.mm(Y - m_ks[k], torch.inverse(W_ks[k])), Y - m_ks[k]).sum(1)
    return quadratic_expectations.transpose(0,1)

def log_expectations_dir(alpha_hat, K):
    log_expectations = torch.zeros(K)
    sum_digamma = torch.digamma(alpha_hat.sum())
    for k in range(K):
        log_expectations[k] = torch.digamma(alpha_hat[k]) - sum_digamma
    return log_expectations

# def vbE_step(alpha_init, alpha_trans, ms, betas, nus, Ws, Y, N, D, K):
#     ## A K by K transition matrix, E is N by K emission matrix, both take log-form
#     log_A = torch.zeros((K, K))
#     quadratic_expectations = quadratic_expectation(nus, Ws, ms, betas, Y, N, D, K)
#
#     log_expectation_lambda = log_expectation_wi(nus, Ws, D, K)
#     log_E = - (D / 2) * torch.log(torch.Tensor([2*math.pi])) - (1 / 2) * log_expectation_lambda - (1 / 2) * quadratic_expectations
#
#     for j in range(K):
#         log_A[j] = log_expectations_dir(alpha_trans[j], K)
#     ## Pi is the initial distribution in qz
#     log_Pi = log_expectations_dir(alpha_init, K)
#     # log_Pi_q = log_Pi - logsumexp(log_Pi)
#     log_Alpha = forward(log_Pi, log_A, log_E, N, K)
#     log_Beta = backward(log_A, log_E, N, K)
#     log_gammas = marginal_posterior(log_Alpha, log_Beta, N, K)
#     log_Eta = joint_posterior(log_Alpha, log_Beta, log_A, log_E, N, K)
#     return log_gammas, log_Eta

def vbE_step(alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    ## A K by K transition matrix, E is N by K emission matrix, both take log-form
    log_A = torch.zeros((K, K))
    quadratic_expectations = quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K)

    log_expectation_lambda = log_expectation_wi(nu_ks, W_ks, D, K)
    log_E = - (D / 2) * torch.log(torch.Tensor([2*math.pi])) - (1 / 2) * log_expectation_lambda - (1 / 2) * quadratic_expectations

    for j in range(K):
        log_A[j] = log_expectations_dir(alpha_trans_hat[j], K)
    ## Pi is the initial distribution in qz
    log_Pi = log_expectations_dir(alpha_init_hat, K)
    # log_Pi_q = log_Pi - logsumexp(log_Pi)
    log_Alpha = forward(log_Pi, log_A, log_E, N, K)
    log_Beta = backward(log_A, log_E, N, K)
    log_gammas = marginal_posterior(log_Alpha, log_Beta, N, K)
    log_Eta = joint_posterior(log_Alpha, log_Beta, log_A, log_E, N, K)
    return log_gammas, log_Eta

def stats(log_gammas, Y, D, K):
    gammas = torch.exp(log_gammas)
    N_ks = gammas.sum(0)
    Y_ks = torch.zeros((K, D))
    S_ks = torch.zeros((K, D, D))
    gammas_expanded = gammas.repeat(D, 1, 1)
    for k in range(K):
        Y_ks[k] = torch.mul(gammas_expanded[:, :, k].transpose(0,1), Y).sum(0) / N_ks[k]
        gammas_expanded2 = gammas_expanded[:, :, k].repeat(D, 1, 1).permute(2, 1, 0)
        Y_diff = Y - Y_ks[k]
        Y_bmm = torch.bmm(Y_diff.unsqueeze(2), Y_diff.unsqueeze(1))
        S_ks[k] = torch.mul(gammas_expanded2, Y_bmm).sum(0) / (N_ks[k])
    return N_ks, Y_ks, S_ks

def vbM_step(log_eta, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, N, D, K):
    eta = torch.exp(log_eta)
    m_ks = torch.zeros((K, D))
    W_ks = torch.zeros((K, D, D))
    cov_ks = torch.zeros((K, D, D))
    alpha_init_hat = alpha_init_0 + N_ks
    nu_ks = nu_0+ N_ks + 1
    beta_ks = beta_0 + N_ks
    alpha_trans_hat = alpha_trans_0 + eta.sum(0)
    for k in range(K):
        m_ks[k] = (beta_0 * m_0[k] + N_ks[k] * Y_ks[k]) / beta_ks[k]
        temp2 = (Y_ks[k] - m_0[k]).view(D, 1)
        W_ks[k] = W_0 + N_ks[k] * S_ks[k] + (beta_0*N_ks[k] / (beta_0 + N_ks[k])) * torch.mul(temp2, temp2.transpose(0, 1))
        cov_ks[k] = W_ks[k] / (nu_ks[k] - D - 1)
    return alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, cov_ks

# def vbM_step(log_eta, alpha_init_0, alpha_trans_0, ms_0, betas_0, nus_0, Ws_0, N_ks, Y_ks, S_ks, N, D, K):
#     eta = torch.exp(log_eta)
#     ms = torch.zeros((K, D))
#     Ws = torch.zeros((K, D, D))
#     covs = torch.zeros((K, D, D))
#     alpha_init = alpha_init_0 + N_ks
#     nus = nus_0+ N_ks + 1
#     betas = betas_0 + N_ks
#     alpha_trans = alpha_trans_0 + eta.sum(0)
#     for k in range(K):
#         ms[k] = (betas_0[k] * ms_0[k] + N_ks[k] * Y_ks[k]) / betas[k]
#         temp2 = (Y_ks[k] - ms_0[k]).view(D, 1)
#         Ws[k] = Ws_0[k] + N_ks[k] * S_ks[k] + (betas_0[k]*N_ks[k] / (betas_0[k] + N_ks[k])) * torch.mul(temp2, temp2.transpose(0, 1))
#         covs[k] = Ws[k] / (nus[k] - D - 1)
#     return alpha_init, alpha_trans, ms, betas, nus, Ws, covs

def log_C(alpha):
    return torch.lgamma(alpha.sum()) - (torch.lgamma(alpha)).sum()

def log_wishart_B(nu, W, D):
    term1 =  (nu / 2) * torch.log(torch.det(W))
    term2 = - (nu * D / 2) * torch.log(torch.Tensor([2]))
    term3 = - (D * (D - 1) / 4) * torch.log(torch.Tensor([math.pi]))
    ds = (nu + 1 - (torch.arange(D).float() + 1)) / 2.0
    term4 = - torch.lgamma(ds).sum()
    return term1 + term2 + term3 + term4

def entropy_iw(nu, W, D, E_log_sigma):
    log_B = log_wishart_B(nu, W, D)
    return - log_B + (nu + D + 1)/2 * E_log_sigma + nu * D / 2

def log_expectations_dir_trans(alpha_trans, K):
    E_log_trans = torch.zeros((K, K))
    for k in range(K):
        E_log_trans[k] = log_expectations_dir(alpha_trans[k], K)
    return E_log_trans

def quad2(a, Sigma, D, K):
    batch_mul = torch.zeros(K)
    for k in range(K):
        ak = a[k].view(D, 1)
        batch_mul[k] = quad(ak, Sigma[k])
    return batch_mul

def kl_niw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K):
    ## input W_ks need to be inversed
    W_ks_inv = torch.zeros((K, D, D))
    for k in range(K):
        W_ks_inv[k] = torch.inverse(W_ks[k])
    E_log_det_sigma = log_expectation_wi(nu_ks, W_ks, D, K)
    trace_W0_Wk_inv = torch.zeros(K)
    entropy = 0.0
    for k in range(K):
        trace_W0_Wk_inv[k] = torch.diag(torch.mm(W_0, W_ks_inv[k])).sum()
        entropy += entropy_iw(nu_ks[k], W_ks[k], D, E_log_det_sigma[k])
    log_B_0 = log_wishart_B(nu_0, torch.inverse(W_0), D)

    E_log_q_phi = (D / 2) * (np.log(beta_ks / (2 * np.pi)) - 1).sum() - entropy
    # E_log_q_phi = - entropy
    quad_mk_m0_Wk = quad2(m_ks - m_0, W_ks_inv, D, K)
    E_log_p_phi = ((1 / 2) * ((np.log(beta_0/(2*np.pi)) - (beta_0 / beta_ks)).sum() * D - torch.mul(nu_ks, quad_mk_m0_Wk).sum() * beta_0)).item()
    E_log_p_phi +=  (K * log_B_0 - ((nu_0 + D + 1) / 2) * E_log_det_sigma.sum() - (1/2) * torch.mul(nu_ks, trace_W0_Wk_inv).sum()).item()
    kl_phi = E_log_q_phi - E_log_p_phi
    return kl_phi

def elbo(log_gammas, log_eta, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, Y, N, D, K):
    gammas = torch.exp(log_gammas)
    eta = torch.exp(log_eta)
    E_log_pi_hat = log_expectations_dir(alpha_init_hat, K)
    E_log_trans = log_expectations_dir_trans(alpha_trans_hat, K)

    ## kl between pz and qz
    E_log_pz = torch.mul(gammas[0], E_log_pi_hat).sum() + torch.mul(E_log_trans.repeat(N-1, 1, 1), eta).sum()
    E_log_qz =torch.mul(log_gammas, gammas).sum()
    kl_z = E_log_qz - E_log_pz

    ## kl between p_pi and q_pi
    kl_pi = log_C(alpha_init_hat) - log_C(alpha_init_0) + torch.mul(alpha_init_hat - alpha_init_0, E_log_pi_hat).sum()

    ## kl between p_A and q_A
    kl_trans = 0.0
    for j in range(K):
        kl_trans += log_C(alpha_trans_hat[j]) - log_C(alpha_trans_0[j]) + torch.mul(alpha_trans_hat[j] - alpha_trans_0[j], E_log_trans[j]).sum()

    # kl between q_phi and p_phi
    # kl_phi = kl_niw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K)
    # print('E_log_qz : %f' % E_q_phi)
    # true_kl_phi = kl_niw_true(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, K, S=500)
    # print('true kl phi : %f' % true_kl_phi)
    kl_phi = kl_niw(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K)
    ##likelihood term
    log_likelihood = 0.0
    for k in range(K):
        E_log_det_sigma_k = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
        Ykmean_diff = (Y_ks[k] - m_ks[k]).view(D, 1)
        log_likelihood += N_ks[k] * (- E_log_det_sigma_k - (D / beta_ks[k]) - nu_ks[k] * (torch.diag(torch.mm(S_ks[k], torch.inverse(W_ks[k]))).sum()) - nu_ks[k] * quad(Ykmean_diff, torch.inverse(W_ks[k])) - D * torch.log(torch.Tensor([2*np.pi])))
    log_likelihood *= 1 / 2
    # print(kl_phi, kl_phi_true)
    # print('NLL : %f, KL_z : %f, KL_pi : %f, KL_trans : %f, KL_phi %f' % (log_likelihood[0][0] , kl_z , kl_pi , kl_trans, kl_phi))
    Elbo = log_likelihood - kl_z - kl_pi - kl_phi - kl_trans
    # Elbo = 0
    return Elbo

def output_transition(alpha_trans_hat):
    A = alpha_trans_hat.data.numpy()
    A = (A.T / A.sum(1)).T
    return A

def plot_results(Y, final_mus, final_covs):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.plot(Y[:,0], Y[:,1], 'ro')
    plot_cov_ellipse(cov=final_covs[0], pos=final_mus[0], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=final_covs[1], pos=final_mus[1], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=final_covs[2], pos=final_mus[2], nstd=2, ax=ax, alpha=0.5)
    plt.show()
