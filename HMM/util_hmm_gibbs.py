import math
import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from scipy.stats import invwishart
import numpy as np
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch
from probtorch.util import log_sum_exp
import time


def pirors(Y, T, D, K):
    ## set up prior
    alpha_init_0 = torch.ones(K)
    # L = torch.ones((K, K))  / (2 *(K-1))
    # alpha_trans_0 = torch.cat((torch.cat((torch.eye(4)*0.5, torch.ones((4, K-4)) * (0.5 / (K-4))), 1),\
    # torch.ones((K-4, K)) * (1.0 / K)), 0)
    alpha_trans_0 = torch.ones((K, K))
    m_0 = torch.FloatTensor([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * (1 / math.sqrt(2))
    beta_0 = 1.0
    nu_0 = 6.0
    W_0 =  (nu_0-D-1) * torch.mm((Y - Y.mean(0)).transpose(0,1), (Y - Y.mean(0))) / (T)
    cov = torch.from_numpy(np.cov(Y.transpose(0,1))).float()
    ## inialize variational distribution as prior
    alpha_init_hat = alpha_init_0
    alpha_trans_hat = alpha_trans_0
    # m_ks = MultivariateNormal(m_0, cov).sample((K,))
    m_ks = torch.FloatTensor([[1, 1], [1, -1], [-1, -1], [-1, 1]])
    beta_ks = (torch.ones(K) * beta_0)
    nu_ks = (torch.ones(K) * nu_0)
    W_ks = W_0.repeat(K, 1, 1)
    return alpha_init_0, alpha_trans_0, m_0, beta_0, nu_0, W_0, alpha_init_hat, alpha_trans_hat, m_ks, beta_ks, nu_ks, W_ks

## K is the number of latent states
## T is the length of the sequence
## Pi is K-length vector representing the initial state probability
## A is the K by K transition matrix, where a_ij = P(z_t = j | z_t-1 = i)
## so sum of each row equals 1.
## E is the T by K vector where e_tk = N(v_t ; mu_k, sigma_k)

def forward(log_Pi, log_A, log_E, T, K):
    log_Alpha = torch.zeros((T, K))
    log_Alpha[0] = log_Pi + log_E[0]
    for t in range(1, T):
        predictor = log_sum_exp(log_Alpha[t-1].repeat(K, 1).transpose(0,1) + log_A, 0)
        log_Alpha[t] = log_E[t] + predictor
    return log_Alpha

def backward(log_A, log_E, T, K):
    log_Beta = torch.zeros((T, K))
    log_Beta[T-1] = torch.zeros(K)
    for t in range(T-1, 0):
        emi_beta_expand = (log_Beta[t] + log_E[t]).repeat(K, 1)
        log_Beta[t-1] = log_sum_exp(emi_beta_expand + log_A, 1)
    return log_Beta

def marginal_posterior(log_Alpha, log_Beta, T, K):
    log_gammas = log_Alpha + log_Beta
    log_gammas = (log_gammas.transpose(0,1) - log_sum_exp(log_gammas, 1)).transpose(0,1)
    return log_gammas

def joint_posterior(log_Alpha, log_Beta, log_A, log_E, T, K):
    log_Eta = torch.zeros((T-1, K, K))
    for t in range(T-1):
        term1 = (log_E[t+1] + log_Beta[t+1]).view(1, K).repeat(K, 1)
        term2 = log_Alpha[t].view(K, 1).repeat(1, K)
        log_joint = term2 + term1 + log_A
        log_Eta[t] = log_joint - log_sum_exp(log_joint)
    return log_Eta

def quad(a, B):
    return torch.mm(torch.mm(a.transpose(0, 1), B), a)

def log_expectation_wi_single(nu, W, D):
    ds = (nu + 1 - (torch.arange(D).float() + 1)) / 2.0
    return  - D * torch.log(torch.Tensor([2])) + torch.log(torch.det(W)) - torch.digamma(ds).sum()

def log_expectation_wi(nu_ks, W_ks, D, K):
    log_expectations = torch.zeros(K)
    for k in range(K):
        log_expectations[k] = log_expectation_wi_single(nu_ks[k], W_ks[k], D)
    return log_expectations

def quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, T, D, K):
    quadratic_expectations = torch.zeros((K, T))
    for k in range(K):
        quadratic_expectations[k] = D / beta_ks[k] + nu_ks[k] * torch.mul(torch.mm(Y - m_ks[k], torch.inverse(W_ks[k])), Y - m_ks[k]).sum(1)
    return quadratic_expectations.transpose(0,1)

def log_expectations_dir(alpha_hat, K):
    log_expectations = torch.zeros(K)
    sum_digamma = torch.digamma(alpha_hat.sum())
    for k in range(K):
        log_expectations[k] = torch.digamma(alpha_hat[k]) - sum_digamma
    return log_expectations

def vbE_step(alpha_init_hat, alpha_trans_hat, nu_ks, W_ks, m_ks, beta_ks, Y, T, D, K):
    ## A K by K transition matrix, E is N by K emission matrix, both take log-form
    log_A = torch.zeros((K, K))
    quadratic_expectations = quadratic_expectation(nu_ks, W_ks, m_ks, beta_ks, Y, T, D, K)

    log_expectation_lambda = log_expectation_wi(nu_ks, W_ks, D, K)
    log_E = - (D / 2) * torch.log(torch.Tensor([2*math.pi])) - (1 / 2) * log_expectation_lambda - (1 / 2) * quadratic_expectations

    for j in range(K):
        log_A[j] = log_expectations_dir(alpha_trans_hat[j], K)
    ## Pi is the initial distribution in qz
    log_Pi = log_expectations_dir(alpha_init_hat, K)
    # log_Pi_q = log_Pi - logsumexp(log_Pi)
    log_Alpha = forward(log_Pi, log_A, log_E, T, K)
    log_Beta = backward(log_A, log_E, T, K)
    log_gammas = marginal_posterior(log_Alpha, log_Beta, T, K)
    log_Eta = joint_posterior(log_Alpha, log_Beta, log_A, log_E, T, K)
    return log_gammas, log_Eta

def stats(Zs, Y, D, K):
    N_ks = Zs.sum(0)
    Y_ks = torch.zeros((K, D))cat
    S_ks = torch.zeros((K, D, D))
    Zs_expanded = Zs.repeat(D, 1, 1)
    for k in range(K):
        Y_ks[k] = torch.mul(Zs_expanded[:, :, k].transpose(0,1), Y).sum(0) / N_ks[k]
        Zs_expanded2 = Zs_expanded[:, :, k].repeat(D, 1, 1).permute(2, 1, 0)
        Y_diff = Y - Y_ks[k]
        Y_bmm = torch.bmm(Y_diff.unsqueeze(2), Y_diff.unsqueeze(1))
        S_ks[k] = torch.mul(Zs_expanded2, Y_bmm).sum(0) / (N_ks[k])
    return N_ks, Y_ks, S_ks

def pairwise(Zs, T):
    return torch.bmm((Zs[:T].unsqueeze(-1), Zs[1:].unsqueeze(1)))

def gibbs_global(Zs, alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, N_ks, Y_ks, S_ks, T, D, K):
    ## Zs is N * K tensor where each row is one-hot encoding of a sample of latent state
    ## sample /pi
    alpha_init_hat = alpha_init_0 + Zs[0]
    pi = Dirichlet(alpha_init_hat).sample()
    ## sample A
    alpha_trans_hat = alpha_trans_0 + pairwise(Zs, T).sum(0)
    A = torch.zeros((K, K))
    for k in range(K):
        A[k] = Dirichlet(alpha_trans_hat[k]).sample()
    ## sample mu_k and Sigma_k
    nu_ks = nu_0 + N_ks + 1
    beta_ks = beta_0 + N_ks
    m_ks = torch.zeros((K, D))
    W_ks = torch.zeros((K, D, D))
    cov_ks = torch.zeros((K, D, D))
    mu_ks = torch.zeros((K, D))
    for k in range(K):
        m_ks[k] = (beta_0 * m_0[k] + N_ks[k] * Y_ks[k]) / beta_ks[k]
        temp2 = (Y_ks[k] - m_0[k]).view(D, 1)
        W_ks[k] = W_0 + N_ks[k] * S_ks[k] + (beta_0*N_ks[k] / (beta_0 + N_ks[k])) * torch.mul(temp2, temp2.transpose(0, 1))
        ## sample mu_k and Sigma_k from posterior
        cov_ks[k] = torch.from_numpy(invwishart.rvs(df=nu_ks[k], scale=W_ks[k].data.numpy()))
        mu_ks[k] = MultivariateNormal(loc=m_ks[k], covariance_matrix=cov_ks[k] / beta_ks[k]).sample()
    return pi, A, mu_ks, cov_ks

def log_joint(alpha_init_0, alpha_trans_0, nu_0, W_0, m_0, beta_0, Zs, pi, A, mu_ks, cov_ks, Y, T, D, K):
    log_joint_prob = 0.0
    ## some vectorization to pick up the k-th global  for each state by one hot encoding
    Y_ll_means = torch.bmm(Zs.unsqueeze(1), mu_ks.repeat(T, 1, 1)).squeeze(1)
    y_ll_covs = torch.mul(Zs.transpose(0,1).repeat(D, D, 1, 1).permute(-1, 2, 1, 0),  cov_ks.repeat(T, 1, 1, 1))
    Ais = torch.bmm(Zs[:T].unsqueeze(1), A.repeat(T-1, 1, 1)).squeeze(1)
    ## start compute LL
    for t in range(T):
        y_joint_prob += MultivariateNormal(Y_ll_means[t], Y_ll_covs[t]).log_prob(Y[t]) # likelihood of obs
        if t == 0:
            y_joint_prob += cat(pi).log_prob(Zs[t]) # z_1 | pi
        else:
            y_joint_prob += cat(Ais[t-1]).log_prob(Zs[t]) # z_t | z_t-1 = j*, A
    for k in range(K):
        y_joint_prob += Dirichlet(alpha_init_0[k]).log_prob(A[k]) ## prior of A
        y_joint_prob += MultivariateNormal(m_0[k], cov_ks[k] / beta_0).log_prob(mu_ks[k])# prior of mu_ks
        y_joint_prob += invwishart.logpdf(cov_ks[k].data.numpy(), nu_0, W_0[k].data.numpy())# prior of cov_ks
    return log_joint_prob

def plot_results(Y, final_mus, final_covs):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.plot(Y[:,0], Y[:,1], 'ro')
    plot_cov_ellipse(cov=final_covs[0], pos=final_mus[0], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=final_covs[1], pos=final_mus[1], nstd=2, ax=ax, alpha=0.5)
    plot_cov_ellipse(cov=final_covs[2], pos=final_mus[2], nstd=2, ax=ax, alpha=0.5)
    plt.show()
