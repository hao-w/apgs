import math
import torch
from torch import logsumexp

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
        predictor = logsumexp(log_Alpha[t-1].repeat(K, 1).transpose(0,1) + log_A, 0)
        log_Alpha[t] = log_E[t] + predictor
    return log_Alpha

def backward(log_A, log_E, T, K):
    log_Beta = torch.zeros((T, K))
    log_Beta[T-1] = torch.zeros(K)
    for t in range(T-1, 0):
        emi_beta_expand = (log_Beta[t] + log_E[t]).repeat(K, 1)
        log_Beta[t-1] = logsumexp(emi_beta_expand + log_A, 1)
    return log_Beta

def marginal_posterior(log_Alpha, log_Beta, T, K):
    log_gammas = log_Alpha + log_Beta
    log_gammas = (log_gammas.transpose(0,1) - logsumexp(log_gammas, 1)).transpose(0,1)
    return log_gammas

def joint_posterior(log_Alpha, log_Beta, log_A, log_E, T, K):
    log_Eta = torch.zeros((T-1, K, K))
    for t in range(T-1):
        term1 = (log_E[t+1] + log_Beta[t+1]).view(1, K).repeat(K, 1)
        term2 = log_Alpha[t].view(K, 1).repeat(1, K)
        log_joint = term2 + term1 + log_A
        log_Eta[t] = log_joint - logsumexp(log_joint.view(K*K, 1), 0)
    return log_Eta
