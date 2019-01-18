import matplotlib.pyplot as plt
import torch
from torch import logsumexp
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical

def smc_lds_batch(Pis, As, mu_ks, cov_ks, Ys, T, D, K, num_particles, batch_size):
    Zs = torch.zeros((num_particles, batch_size, T, K))
    log_weights = torch.zeros((num_particles, batch_size, T))
    log_normalizers = torch.zeros(batch_size).float()
    for t in range(T):
        if t == 0:
            sample_zs = cat(Pis).sample((num_particles, batch_size))
        else:
            # reweights B-by-smc tensor
            reweights = torch.exp(log_weights[:, :, t-1] - logsumexp(log_weights[:, :, t-1], dim=0)).transpose(0,1)
            ## ancesters smc-by-B tensor, and expand it to B-smc-T-K index matrix for gather function
            ancesters = Categorical(reweights).sample((num_particles,)).transpose(0,1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, K)
            ## resampling and swap the two axises twice which makes it back to original
            Zs = torch.gather(Zs.transpose(0,1), 1, ancesters).transpose(0,1)
            labels = Zs[:, :, t-1, :].nonzero()
            sample_zs = cat(As[labels[:, 1], labels[:, -1]]).sample().view(num_particles, batch_size, K)
        Zs[:, :, t, :] = sample_zs
        labels = Zs[:, :, t, :].nonzero()[:, -1]
        likelihoods = MultivariateNormal(mu_ks[labels].view(num_particles, batch_size, D), cov_ks[labels].view(num_particles, batch_size, D, D)).log_prob(Ys[:, t])
        log_weights[:, :, t] = likelihoods
        log_normalizers = log_normalizers + (logsumexp(log_weights[:, :, t], dim=0) - torch.log(torch.FloatTensor([num_particles])))
    return Zs.transpose(0,1), log_weights, log_normalizers

def smc_resamplings(Zs, log_weights, batch_size):
    reweights = torch.exp(log_weights[:, :, -1] - logsumexp(log_weights[:, :, -1], dim=0)).transpose(0,1)
    ancesters = Categorical(reweights).sample()
    return Zs[torch.arange(batch_size), ancesters]

def smc_log_joints(Z, Pis, As, mu_ks, cov_ks, Ys, T, D, K, batch_size):
    log_joints = torch.zeros(batch_size).float()
    for t in range(T):
        labels = Z[:, t].nonzero()[:, -1]
        likelihoods = MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Ys[:, t])
        log_joints = log_joints + likelihoods
        if t == 0:
            log_joints = log_joints + cat(Pis).log_prob(Z[:, t])
        else:
            label_prev = Z[:, t-1].nonzero()
            log_joints = log_joints + cat(As[label_prev[:, 0], label_prev[:, -1]]).log_prob(Z[:, t])
    return log_joints
