import matplotlib.pyplot as plt
import torch
from torch import logsumexp
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical



def csmc_hmm_v(Z_ret, Pi, As, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws):
    ##Z_ret is rws-by-T-by-K
    Zs = torch.zeros((num_particles_rws, num_particles_smc, T, K))
    log_weights = torch.zeros((num_particles_rws, num_particles_smc, T))
    log_normalizers = torch.zeros(num_particles_rws).float()
    for t in range(T):
        if t == 0:
            Zs[:, -1, t, :] = Z_ret[:, t, :]
            sample_zs = cat(Pi).sample((num_particles_rws, num_particles_smc-1))
            Zs[:, :-1, t, :] = sample_zs
            labels = Zs[:, :, t, :].nonzero()[:, -1]
            likelihoods = MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Y[t]).view(num_particles_rws, num_particles_smc)
            log_weights[:, :, t] = likelihoods
        else:
            reweight = torch.exp(log_weights[:, :, t-1] - logsumexp(log_weights[:, :, t-1], dim=1).unsqueeze(1))
            inds = Categorical(reweight).sample((num_particles_smc-1,)).transpose(0,1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, K)
            Zs[:, :-1, :, :] = torch.gather(Zs, 1, inds)
            labels = Zs[:, :-1, t-1, :].nonzero()
            sample_zs = cat(As[labels[:,0], labels[:, -1]]).sample().view(num_particles_rws, num_particles_smc-1, K)
            Zs[:, :-1, t, :] = sample_zs
            Zs[:, -1, t, :] = Z_ret[:, t, :]
            labels = Zs[:, :, t, :].nonzero()[:, -1]
            likelihoods = MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Y[t]).view(num_particles_rws, num_particles_smc)
            log_weights[:, :, t] = likelihoods
        log_normalizers = log_normalizers + (logsumexp(log_weights[:, :, t], dim=1) - torch.log(torch.FloatTensor([num_particles_smc])) )
    return Zs, log_weights, log_normalizers

def smc_hmm_v(Pi, As, mu_ks, cov_ks, Y, T, D, K, num_particles_smc, num_particles_rws):
    Zs = torch.zeros((num_particles_rws, num_particles_smc, T, K))
    log_weights = torch.zeros((num_particles_rws, num_particles_smc, T))
    log_normalizers = torch.zeros(num_particles_rws).float()
    for t in range(T):
        if t == 0:
            sample_zs = cat(Pi).sample((num_particles_rws, num_particles_smc))
            Zs[:, :, t, :] = sample_zs
            labels = sample_zs.nonzero()[:, -1]
            likelihoods = MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Y[t]).view(num_particles_rws, num_particles_smc)
            log_weights[:, :, t] = likelihoods
        else:
            reweight = torch.exp(log_weights[:, :, t-1] - logsumexp(log_weights[:, :, t-1], dim=1).unsqueeze(1))
            inds = Categorical(reweight).sample((num_particles_smc,)).transpose(0,1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, K)
            Zs = torch.gather(Zs, 1, inds)
            labels = Zs[:, :, t-1, :].nonzero()
            sample_zs = cat(As[labels[:,0], labels[:, -1]]).sample().view(num_particles_rws, num_particles_smc, K)
            Zs[:, :, t, :] = sample_zs
            labels = sample_zs.nonzero()[:, -1]
            likelihoods = MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Y[t]).view(num_particles_rws, num_particles_smc)
            log_weights[:, :, t] = likelihoods
        log_normalizers = log_normalizers + (logsumexp(log_weights[:, :, t], dim=1) - torch.log(torch.FloatTensor([num_particles_smc])) )
    return Zs, log_weights, log_normalizers

def resampling_smc_v(Zs, log_weights, num_particles_rws):
    reweight = torch.exp(log_weights[:, :, -1] - logsumexp(log_weights[:, :, -1], dim=1).unsqueeze(1))
    inds = Categorical(reweight).sample()
    Z_ret = Zs[torch.arange(num_particles_rws), inds]
    return Z_ret

def log_joint_smc_v(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_rws):
    log_joints = torch.zeros(num_particles_rws).float()
    labels = Z_ret.nonzero()
    labels_trans = (labels.view(num_particles_rws, T, -1)[:, :-1, :]).contiguous().view(num_particles_rws*(T-1), 3)
    Z_ret_trans = Z_ret[:, 1:, :].contiguous().view(num_particles_rws*(T-1), -1)
    Ys = Y.repeat(num_particles_rws, 1, 1).view(num_particles_rws*T, D)
    log_joints = log_joints + (MultivariateNormal(mu_ks[labels[:,-1]], cov_ks[labels[:,-1]]).log_prob(Ys)).view(num_particles_rws, T).sum(1)
    log_joints = log_joints + cat(Pi).log_prob(Z_ret[:, 0, :])
    log_joints = log_joints + (cat(A_samples[labels_trans[:,0], labels_trans[:,-1]]).log_prob(Z_ret_trans).view(num_particles_rws, T-1).sum(1))
    return log_joints
