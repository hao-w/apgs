import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch
from probtorch.util import log_sum_exp
from bokeh.plotting import figure, output_notebook, show
from util_gmm import log_expectation_dir, log_expectation_wi

def logZ(nu, W, beta, D):
    term1 = - (D * (D + 1) / 2) * torch.log(torch.Tensor([2*math.pi]))
    term2 = - (nu * D / 2) * torch.log(torch.Tensor([math.pi]))
    term3 = - (D / 2) * torch.log(torch.Tensor([beta]))
    term4 = - (nu / 2) * torch.log(torch.det(W))
    term5 = (D * (D - 1) / 4) * torch.log(torch.Tensor([math.pi]))
    ds = (nu + 1 - (torch.arange(D).float() + 1)) / 2.0
    term6 = torch.lgamma(ds).sum()
    return term1 + term2 + term3 + term4 + term5 + term6

def expectedstats(nu_ks, W_ks, m_ks, beta_ks, D, K):
    W_ks_inv = torch.zeros((K, D, D))
    E_log_W_ks = log_expectation_wi(nu_ks, W_ks, D, K)
    mean_mu_ks = torch.mul(nu_ks.repeat(D, D, 1).transpose(-1,0), W_ks_inv)
    for k in range(K):
        W_ks_inv[k] = torch.inverse(W_ks[k])
    # K * D
    niw_stat1 = torch.bmm(mean_mu_ks, m_ks.unsqueeze(-1)).squeeze(-1)
    # K * D * D
    niw_stat2 = - (1 / 2) * mean_mu_ks
    # K
    niw_stat3 = D / beta_ks + torch.bmm(torch.bmm(m_ks.unsqueeze(1), mean_mu_ks), m_ks.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    niw_stat3 *= -1
    # K
    niw_stat4 = (D / 2) * torch.log(torch.Tensor([2*math.pi])) + (1/2) * E_log_W_ks
    niw_stat4 *= -1
    return niw_stat1, niw_stat2, niw_stat3, niw_stat4

def param_to_natparam(nu_ks, W_ks, m_ks, beta_ks, D, K):
    # K * D
    niw_eta1 = (m_ks.transpose(1,0) * beta_ks).transpose(1,0)
    # K * D * D
    niw_eta2 = torch.bmm(niw_eta1.unsqueeze(-1), m_ks.unsqueeze(1)) + W_ks
    # K
    niw_eta3 = beta_ks
    # K
    niw_eta4 = nu_ks + D + 2
    return niw_eta1, niw_eta2, niw_eta3, niw_eta4


def kl_niw_2(nu_0, W_0, m_0, beta_0, nu_ks, W_ks, m_ks, beta_ks, D, K):

    niw_stat1, niw_stat2, niw_stat3, niw_stat4 = expectedstats(nu_ks, W_ks, m_ks, beta_ks, D, K)
    q_niw_eta1, q_niw_eta2, q_niw_eta3, q_niw_eta4 = param_to_natparam(nu_ks, W_ks, m_ks, beta_ks, D, K)
    p_niw_eta1, p_niw_eta2, p_niw_eta3, p_niw_eta4 = param_to_natparam(nu_0 * torch.ones(K), W_0.repeat(K, 1, 1), m_0.repeat(K,1), beta_0 * torch.ones(K), D, K)
    eta1_diff = q_niw_eta1 - p_niw_eta1
    eta2_diff = q_niw_eta2 - p_niw_eta2
    eta3_diff = q_niw_eta3 - p_niw_eta3
    eta4_diff = q_niw_eta4 - p_niw_eta4
    inner1 = torch.bmm(eta1_diff.unsqueeze(1), niw_stat1.unsqueeze(-1)).squeeze(-1).squeeze(-1).sum()
    inner2 = torch.diagonal(torch.bmm(eta2_diff.transpose(-1,-2), niw_stat2.transpose(-1,-2)), offset=0, dim1=-2, dim2=-1).sum()
    inner3 = torch.mul(eta3_diff, niw_stat3).sum()
    inner4 = torch.mul(eta4_diff, niw_stat4).sum()
    p_logZ = logZ(nu_0, W_0, beta_0, D) * K
    q_logZ = 0.0
    for k in range(K):
        q_logZ += logZ(nu_ks[k], W_ks[k], beta_ks[k], D)
    kl_phi = p_logZ - q_logZ + inner1 + inner2 + inner3 + inner4
    return kl_phi
