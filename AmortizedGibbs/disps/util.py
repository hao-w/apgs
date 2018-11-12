import torch
from torch import logsumexp
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical

def save_params(KLs, EUBOs, ELBOs, PATH_ENC):
    with open(PATH_ENC + 'EUBOs.txt', 'w+') as feubo:
        for eubo in EUBOs:
            feubo.write("%s\n" % eubo)
    with open(PATH_ENC + 'KLs.txt', 'w+') as fkl:
        for kl in KLs:
            fkl.write("%s\n" % kl)
    with open(PATH_ENC + 'ELBOs.txt', 'w+') as felbo:
        for elbo in ELBOs:
            fess.write("%s\n" % elbo)
    feubo.close()
    felbos.close()
    fkls.close()

def initial_trans_prior(K):
    alpha_trans_0 = torch.ones((K, K))
    return alpha_trans_0

def log_q_hmm(latents_dirs, A_samples):
    log_q = Dirichlet(latents_dirs).log_prob(A_samples)
    return log_q.sum()

def pairwise(Zs, T):
    return torch.bmm(Zs[:T-1].unsqueeze(-1), Zs[1:].unsqueeze(1))

def log_joint(alpha_trans_0, Zs, Pi, A, mu_ks, cov_ks, Y, T, D, K):
    log_joint_prob = torch.zeros(1).float()
    labels = Zs.nonzero()[:, 1]
    log_joint_prob += (MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Y)).sum() # likelihood of obs
    log_joint_prob += cat(Pi).log_prob(Zs[0]) # z_1 | pi
    log_joint_prob += (cat(A[labels[:-1]]).log_prob(Zs[1:])).sum()
    log_joint_prob += (Dirichlet(alpha_trans_0).log_prob(A)).sum() ## prior of A
    return log_joint_prob

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