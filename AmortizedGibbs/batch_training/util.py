import torch
from torch import logsumexp
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical

def save_params(EUBOs, ELBOs, PATH_ENC):
    with open(PATH_ENC + 'EUBOs.txt', 'w+') as feubo:
        for eubo in EUBOs:
            feubo.write("%s\n" % eubo)
    with open(PATH_ENC + 'ELBOs.txt', 'w+') as felbo:
        for elbo in ELBOs:
            felbo.write("%s\n" % elbo)
    feubo.close()
    felbo.close()

def initial_trans_prior(K):
    alpha_trans_0 = torch.ones((K, K))
    for k in range(K):
        alpha_trans_0[k, k] = 4.0
    return alpha_trans_0

def initial_trans(alpha_trans_0, K, batch_size):
    As = torch.zeros((batch_size, K, K)).float()
    for k in range(K):
        As[:, k, :] = Dirichlet(alpha_trans_0[k]).sample((batch_size,))
    return As

def log_qs(variational, As):
    ## latents_dirs B-K-K, and As also B-K-K
    ## returns B-length vector
    log_q = Dirichlet(variational).log_prob(As)
    return log_q.sum(1)

def log_joints(alpha_trans_0, Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size):
    log_probs = torch.zeros(batch_size).float()
    labels = Z.nonzero()
    log_probs = log_probs + (MultivariateNormal(mu_ks[labels[:, -1]].view(batch_size, T, D), cov_ks[labels[:, -1]].view(batch_size, T, D, D)).log_prob(Ys)).sum(1) # likelihood of obs
    log_probs = log_probs + cat(Pi).log_prob(Z[:, 0]) # z_1 | pi
    inds = labels.view(batch_size, T, K)[:, :-1, :].contiguous().view(batch_size * (T-1), K)

    log_probs = log_probs + (cat(As[inds[:, 0], inds[:, -1]].view(batch_size, T-1, K)).log_prob(Z[:, 1:])).sum(1)
    log_probs = log_probs + (Dirichlet(alpha_trans_0).log_prob(As)).sum(1) ## prior of A
    return log_probs

def kl_dirichlets(posterior, variational, K):
    kl = 0.0
    for k in range(K):
        kl += kl_dirichlet(posterior[k], variational[k])
    return kl

def kl_dirichlet(alpha1, alpha2):
    A = torch.lgamma(alpha1.sum()) - torch.lgamma(alpha2.sum())
    B = (torch.lgamma(alpha1) - torch.lgamma(alpha2)).sum()
    C = (torch.mul(alpha1 - alpha2, torch.digamma(alpha1) - torch.digamma(alpha1.sum()))).sum()
    kl = A - B + C
    return kl

def BCE(x_mean, x, EPS=1e-9):
    return - (torch.log(x_mean + EPS) * x +
              torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)
