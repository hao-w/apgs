import math
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from numbers import Number
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch
from probtorch.util import log_sum_exp
from bokeh.plotting import figure, output_notebook, show
from niw import *
## compute the long quadratic-like term when updating qz
def quadratic_expectation_2(nu_ks, W_ks, m_ks, beta_ks, qx_mu, qx_sigma, batch_size, D, K):
    quadratic_expectations = torch.zeros((K, batch_size))
    for k in range(K):
        W_k_inv = torch.inverse(W_ks[k])
        trace_k = nu_ks[k].item() * torch.diagonal(torch.bmm(W_k_inv.repeat(batch_size, 1, 1), qx_sigma), offset=0, dim1=-2, dim2=-1).sum(1)
        quad_k = nu_ks[k].item() * torch.mul(torch.mm(qx_mu - m_ks[k], W_k_inv), qx_mu - m_ks[k]).sum(1)
        quadratic_expectations[k] = D / beta_ks[k].item() + trace_k + quad_k
    return quadratic_expectations.transpose(0,1)

## compute natural parameters from global parameters part (similar to VBM step)
def gaussian_global(log_gammas, nu_ks, W_ks, m_ks, N, D, K):
    gammas_expanded = torch.exp(log_gammas).repeat(D, 1, 1)
    # N * K * D
    gammas_expanded1 = gammas_expanded.transpose(0, 1).transpose(1, 2)
    # N * K * D * D
    gammas_expanded2 = gammas_expanded.repeat(D, 1, 1, 1).transpose(0, 2).transpose(1, 3)
    eta2 = torch.zeros((K, D, D))
    eta1 = torch.zeros((K, D))
    for k in range(K):
        eta1[k] = (nu_ks[k].item() * torch.mm(torch.inverse(W_ks[k]), m_ks[k].view(D, 1))).squeeze(1)
        eta2[k] = (-1 / 2) * nu_ks[k].item() * torch.inverse(W_ks[k])
    eta1_expanded = eta1.repeat(N, 1, 1)
    eta2_expanded = eta2.repeat(N, 1, 1, 1)
    batch_eta1 = torch.mul(eta1_expanded, gammas_expanded1).sum(1)
    batch_eta2 = torch.mul(eta2_expanded, gammas_expanded2).sum(1)
    return batch_eta1, batch_eta2

max_iter = 1
## iterativly optimize qx and qz until converging
def local_optimal(nn_eta1, nn_eta2, log_gammas, alpha_hat, nu_ks, W_ks, m_ks, beta_ks, D, K):
    batch_size = nn_eta2.squeeze(0).shape[0]
    full_nn_eta2 = torch.zeros((batch_size, D, D))
    full_nn_eta2[:, torch.arange(D).long(), torch.arange(D).long()] = nn_eta2.squeeze(0)
    nn_eta1 = nn_eta1.squeeze(0)

    for i in range(max_iter):
        ## update qx
        batch_eta1, batch_eta2 = gaussian_global(log_gammas, nu_ks, W_ks, m_ks, batch_size, D, K)
        ## N * D
        qx_eta1 = batch_eta1 + nn_eta1
        ## N * D * D
        qx_eta2 = batch_eta2 + full_nn_eta2
        ## compute
        ## update qz, similar to VBE step in VBEM-GMM
        log_gammas = label_update(qx_eta1, qx_eta2, alpha_hat, nu_ks, W_ks, m_ks, beta_ks, batch_size, D, K)

        ## see if converge
        if i + 1 == max_iter:
            print('iteration limit reached')
    return batch_eta1, batch_eta2
## update qz and return log_gammas
def label_update(qx_eta1, qx_eta2, alpha_hat, nu_ks, W_ks, m_ks, beta_ks, batch_size, D, K):
    E_log_pi = log_expectation_dir(alpha_hat, K)
    E_log_sigma = log_expectation_wi(nu_ks, W_ks, D, K)

    ## qx_sigma is N * D * D, qx_mu is N * D

    qx_sigma = torch.zeros((batch_size, D, D))
    for n in range(batch_size):
        qx_sigma[n] = (-1 / 2) * torch.inverse(qx_eta2[n])
    qx_mu = torch.bmm(qx_sigma, qx_eta1.unsqueeze(-1)).squeeze(-1)

    expectedquad = quadratic_expectation_2(nu_ks, W_ks, m_ks, beta_ks, qx_mu, qx_sigma, batch_size, D, K)
    for k in range(K):
        log_rhos = E_log_pi - (D / 2) * torch.log(torch.Tensor([2*math.pi])) - (1 / 2) * E_log_sigma - (1 / 2) * expectedquad
    log_gammas = (log_rhos.transpose(0,1) - log_sum_exp(log_rhos, 1)).transpose(0,1)
    return log_gammas


def stats_2(log_gammas, qx_mu, qx_sigma, D, K):
    gammas = torch.exp(log_gammas)# optimizer.zero_grad()
    N = gammas.shape[0]
    N_ks = gammas.sum(0)
    qx_mu_expanded = qx_mu.repeat(K, 1, 1)
    gammas_expanded1 = gammas.repeat(D, 1, 1)
    gammas_expanded2 = gammas.repeat(D, D, 1, 1)
    ## D * N * K
    qx_mu_ks = torch.mul(qx_mu_expanded.transpose(-1, 0), gammas_expanded1).sum(1) / N_ks
    qx_mu_ks = qx_mu_ks.transpose(1,0)
    diff = (qx_mu_expanded.transpose(1, 0) - qx_mu_ks).view(N*K, D)
    NS_ks = torch.mul(torch.bmm(diff.unsqueeze(-1), diff.unsqueeze(1)).view(N, K, D, D), gammas_expanded2.permute(2, -1, 0, 1)).sum(0)
    ## K * N * D * D
    nk_qx_sigma_ks = torch.mul(qx_sigma.repeat(K, 1, 1, 1), gammas_expanded2.permute(-1, 2, 0, 1)).sum(1)
    return N_ks, qx_mu_ks, NS_ks, nk_qx_sigma_ks

def global_optimal(alpha_0, nu_0, W_0, m_0, beta_0, N_ks, qx_mu_ks, NS_ks, nk_qx_sigma_ks, batch_size, D, K):
    alpha_hat = alpha_0 + N_ks
    nu_ks = nu_0+ N_ks + 1
    beta_ks = beta_0 + N_ks
    m_ks = (((qx_mu_ks.transpose(1, 0) * N_ks).transpose(1, 0) + beta_0 * m_0).transpose(1, 0) / beta_ks).transpose(1,0)

    diff = qx_mu_ks - m_0
    W_ks = W_0.repeat(K, 1, 1) + NS_ks + nk_qx_sigma_ks + torch.mul(torch.bmm(diff.unsqueeze(-1), diff.unsqueeze(1)), (beta_0 * N_ks / beta_ks).repeat(D, D, 1).transpose(-1, 0))
    cov_ks = torch.mul(W_ks, 1 / (nu_ks - D - 1).repeat(D, D, 1).transpose(-1, 0))
    return alpha_hat, nu_ks, W_ks, m_ks, beta_ks, cov_ks

def log_like(q, p, sample_dim=None, batch_dim=None, log_weights=None,
             size_average=True, reduce=True):

    x = [n for n in p.conditioned() if n not in q]
    objective = p.log_joint(sample_dim, batch_dim, x)
    if sample_dim is not None:
        if log_weights is None:
            log_weights = q.log_joint(sample_dim, batch_dim, q.conditioned())
        if isinstance(log_weights, Number):
            objective = objective.mean(0)
        else:
            weights = softmax(log_weights, 0)
            objective = (weights * objective).sum(0)
    if reduce:
        objective = objective.mean() if size_average else objective.sum()
    return objective

def log_C(alpha):
    return torch.lgamma(alpha.sum()) - (torch.lgamma(alpha)).sum()

def E_kl_qx_px(nu_ks, W_ks, m_ks, beta_ks, log_gammas, qx_mu, qx_sigma, batch_size, D, K):
    gammas = torch.exp(log_gammas)
    E_log_sigma = log_expectation_wi(nu_ks, W_ks, D, K)
    expectedquad = quadratic_expectation_2(nu_ks, W_ks, m_ks, beta_ks, qx_mu, qx_sigma, batch_size, D, K)
    for k in range(K):
        log_rhos = - (D / 2) * torch.log(torch.Tensor([2*math.pi])) - (1 / 2) * E_log_sigma - (1 / 2) * expectedquad
    logpx = torch.mul(gammas, log_rhos).sum()
    logqx = - MultivariateNormal(qx_mu, qx_sigma).entropy().sum()
    return logqx - logpx

def elbo_nn(qx_mu, qx_sigma, log_gammas, alpha_0, nu_0, W_0, m_0, beta_0, alpha_hat, nu_ks, W_ks, m_ks, beta_ks, batch_size, D, K, q, p, sample_dim=None, batch_dim=None, log_weights=None, size_average=True, reduce=True):
    gammas = torch.exp(log_gammas)
    E_log_pi_hat = log_expectation_dir(alpha_hat, K)
    E_log_pi_hat_expanded = E_log_pi_hat.repeat(batch_size, 1)
    ## kl between p_pi and q_pi
    kl_pi = log_C(alpha_hat) - log_C(alpha_0) + torch.mul(alpha_hat - alpha_0, E_log_pi_hat).sum()
    ## kl between pz and qz
    kl_z = torch.mul(log_gammas - E_log_pi_hat_expanded, gammas).sum()
    ## kl between q_mu_k, q_sigma_k and p_mu_k, p_sigma_k
    kl_phi = kl_niw_2(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K)
    ## kl between qx and px|z,mu_k,p_sigma_k
    kl_x = E_kl_qx_px(nu_ks, W_ks, m_ks, beta_ks, log_gammas, qx_mu, qx_sigma, batch_size, D, K)
    ## log likelihood
    log_weights = q.log_joint(sample_dim, batch_dim, q.conditioned())
    loglikelihood = log_like(q, p, sample_dim, batch_dim, log_weights,
                     size_average=size_average, reduce=reduce)
    elbo = loglikelihood - kl_pi.cuda() - kl_z.cuda() - kl_phi.float().cuda() - kl_x.cuda()
    return elbo
