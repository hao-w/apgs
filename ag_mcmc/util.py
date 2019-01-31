import torch
from torch import logsumexp
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical

def save_params(EUBOs, ELBOs, ESSs, PATH_ENC):
    with open(PATH_ENC + 'ELBOs.txt', 'w+') as felbo:
        for elbo in ELBOs:
            felbo.write("%s\n" % elbo)
    with open(PATH_ENC + 'EUBOs.txt', 'w+') as feubo:
        for eubo in EUBOs:
            feubo.write("%s\n" % eubo)
            
    with open(PATH_ENC + 'ESSs.txt', 'w+') as fess:
        for ess in ESSs:
            fess.write("%s\n" % ess)
    felbo.close()
    feubo.close()
    fess.close()
    
    
def conj_posterior(priors, Zs, T, K, batch_size):
    """
    Zs is B-T-K tensor
    returns a B-K-K tensor
    """
    prevs = Zs[:, :T-1, :].contiguous().view(batch_size*(T-1), K).unsqueeze(-1)
    nexts = Zs[:, 1:, :].contiguous().view(batch_size*(T-1), K).unsqueeze(1)

    return priors + torch.bmm(prevs, nexts).contiguous().view(batch_size, T-1, K, K).sum(1)


def pairwise(Zs, T):
    return torch.bmm(Zs[:T-1].unsqueeze(-1), Zs[1:].unsqueeze(1))

def initial_trans_prior(K):
    alpha_trans_0 = torch.ones((K, K)).float()
    for k in range(K):
        alpha_trans_0[k, k] = 5.0
    return alpha_trans_0

def initial_trans(prior, K, batch_size):
    As = torch.zeros((batch_size, K, K)).float()
    for k in range(K):
        As[:, k, :] = Dirichlet(prior[k]).sample((batch_size,))
    return As

def log_qs(variational, As):
    ## latents_dirs B-K-K, and As also B-K-K
    ## returns B-length vector
    log_q = Dirichlet(variational).log_prob(As)
    return log_q.sum(1)

def log_joints_ball(alpha_trans_0, Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size):
    log_probs = torch.zeros(batch_size).float()
    labels = Z.nonzero()
    log_probs = log_probs + (MultivariateNormal(mu_ks[labels[:, 0], labels[:, -1]].view(batch_size, T, D), cov_ks[labels[:,0], labels[:, -1]].view(batch_size, T, D, D)).log_prob(Ys)).sum(1) # likelihood of obs
    log_probs = log_probs + cat(Pi).log_prob(Z[:, 0]) # z_1 | pi
    inds = labels.view(batch_size, T, 3)[:, :-1, :].contiguous().view(batch_size * (T-1), 3)
    log_probs = log_probs + (cat(As[inds[:, 0], inds[:, -1]].view(batch_size, T-1, K)).log_prob(Z[:, 1:])).sum(1)
    log_probs = log_probs + (Dirichlet(alpha_trans_0).log_prob(As)).sum(1) ## prior of A
    return log_probs

def log_joints(prior, Z, Pi, As, mus, covs, Ys, T, D, K, batch_size):
    log_probs = torch.zeros(batch_size).float()
    labels = Z.nonzero()
    log_probs = log_probs + (MultivariateNormal(mus[labels[:, -1]].view(batch_size, T, D), covs[labels[:, -1]].view(batch_size, T, D, D)).log_prob(Ys)).sum(1) # likelihood of obs
    log_probs = log_probs + cat(Pi).log_prob(Z[:, 0]) # z_1 | pi
    inds = labels.view(batch_size, T, 3)[:, :-1, :].contiguous().view(batch_size * (T-1), 3)
    log_probs = log_probs + (cat(As[inds[:, 0], inds[:, -1]].view(batch_size, T-1, K)).log_prob(Z[:, 1:])).sum(1)
    log_probs = log_probs + (Dirichlet(prior).log_prob(As)).sum(1) ## prior of A
    return log_probs

def kl_dirichlets(p, q):
    sum_p_concentration = p.sum(-1)
    sum_q_concentration = q.sum(-1)
    t1 = sum_p_concentration.lgamma() - sum_q_concentration.lgamma()
    t2 = (p.lgamma() - q.lgamma()).sum(-1)
    t3 = p - q
    t4 = p.digamma() - sum_p_concentration.digamma().unsqueeze(-1)
    return t1 - t2 + (t3 * t4).sum(-1)

def BCE(x_mean, x, EPS=1e-9):
    return - (torch.log(x_mean + EPS) * x +
              torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)
