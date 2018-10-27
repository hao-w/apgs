import matplotlib.pyplot as plt
import torch
from torch import logsumexp
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical

def csmc_hmm(Z_ret, Pi, A, mu_ks, cov_ks, Y, T, D, K, num_particles_smc):
    Zs = torch.zeros((num_particles_smc, T, K))
    log_weights = torch.zeros((num_particles_smc, T))
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    log_normalizer = torch.zeros(1).float()
    for t in range(T):
        if t == 0:
            Zs[-1, t] = Z_ret[t]
            label = Z_ret[t].nonzero().item()
            sample_zs = cat(Pi).sample((num_particles_smc-1,))
            Zs[:-1, t, :] = sample_zs
            labels = Zs[:, t, :].nonzero()[:, 1]
            likelihoods = MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Y[t])
            log_weights[:, t] = likelihoods
        else:
            reweight = torch.exp(log_weights[:, t-1] - logsumexp(log_weights[:, t-1]))
            ancesters = Categorical(reweight).sample((num_particles_smc-1,))
            Zs[:-1] = Zs[ancesters]
            Zs[-1, t] = Z_ret[t]
            labels = Zs[:-1, t-1, :].nonzero()[:, 1]
            sample_zs = cat(A[labels]).sample()
            Zs[:-1, t, :] = sample_zs
            labels = Zs[:, t, :].nonzero()[:, 1]
            likelihoods = MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Y[t])
            log_weights[:, t] = likelihoods
        log_normalizer += logsumexp(log_weights[:, t]) - torch.log(torch.FloatTensor([num_particles_smc]))
    return Zs, log_weights, log_normalizer

# def csmc_hmm(Z_ret, Pi, A, mu_ks, cov_ks, Y, T, D, K, num_particles=1):
#     Zs = torch.zeros((num_particles, T, K))
#     log_weights = torch.zeros((num_particles, T))
#     decode_onehot = torch.arange(K).float().unsqueeze(-1)
#     log_normalizer = torch.zeros(1).float()
#
#     for t in range(T):
#         if t == 0:
#             Zs[-1, t] = Z_ret[t]
#             label = Z_ret[t].nonzero().item()
#             likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
#             log_weights[-1, t] = likelihood
#             for n in range(num_particles-1):
#                 sample_z = cat(Pi).sample()
#                 Zs[n, t] = sample_z
#                 label = sample_z.nonzero().item()
#                 likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
#                 log_weights[n, t] = likelihood
#
#         else:
#             reweight = torch.exp(log_weights[:, t-1] - logsumexp(log_weights[:, t-1]))
#             ancesters = Categorical(reweight).sample((num_particles-1,))
#             Zs[:-1] = Zs[ancesters]
#             Zs[-1, t] = Z_ret[t]
#             label = Z_ret[t].nonzero().item()
#             likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
#             log_weights[-1, t] = likelihood
#             for n in range(num_particles-1):
#                 label = torch.mm(Zs[n, t-1].unsqueeze(0), decode_onehot).int().item()
#                 sample_z = cat(A[label]).sample()
#                 Zs[n, t] = sample_z
#                 label = sample_z.nonzero().item()
#                 likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
#                 log_weights[n, t] = likelihood
#         log_normalizer += logsumexp(log_weights[:, t]) - torch.log(torch.FloatTensor([num_particles]))
#     return Zs, log_weights, log_normalizer



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

def smc_hmm(Pi, A, mu_ks, cov_ks, Y, T, D, K, num_particles=1):
    Zs = torch.zeros((num_particles, T, K))
    log_weights = torch.zeros((num_particles, T))
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    log_normalizer = torch.zeros(1).float()
    for t in range(T):
        if t == 0:
            for n in range(num_particles):
                sample_z = cat(Pi).sample()
                Zs[n, t] = sample_z
                label = sample_z.nonzero().item()
                likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
                log_weights[n, t] = likelihood
        else:
            ## resampling
            reweight = torch.exp(log_weights[:, t-1] - logsumexp(log_weights[:, t-1], dim=0))
            ancesters = Categorical(reweight).sample((num_particles,))
            Zs = Zs[ancesters]
            for n in range(num_particles):
                label = torch.mm(Zs[n, t-1].unsqueeze(0), decode_onehot).int().item()
                sample_z = cat(A[label]).sample()
                Zs[n, t] = sample_z
                label = sample_z.nonzero().item()
                likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
                log_weights[n, t] = likelihood
        log_normalizer += logsumexp(log_weights[:, t], dim=0) - torch.log(torch.FloatTensor([num_particles]))
        #
        # for n in range(num_particles):
        #     path = torch.mm(Zs[n, :t+1], decode_onehot).data.numpy()
        #     plt.plot(path, 'r-o')
        # plt.show()

    return Zs, log_weights, log_normalizer

def resampling_smc(Zs, log_weights):
    reweight = log_weights[:, -1] - logsumexp(log_weights[:, -1])
    ancester = Categorical(reweight).sample().item()
    return Zs[ancester]

# def log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K):
#     log_joint = torch.zeros(1).float()
#     decode_onehot = torch.arange(K).float().unsqueeze(-1)
#     for t in range(T):
#         label = Z_ret[t].nonzero().item()
#         likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
#         log_joint += likelihood
#         if t == 0:
#             log_joint += cat(Pi).log_prob(Z_ret[t])
#         else:
#             label_prev = Z_ret[t-1].nonzero().item()
#             log_joint += cat(A_samples[label_prev]).log_prob(Z_ret[t])
#     return log_joint

def log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K):
    log_joint = torch.zeros(1).float()
    labels = Z_ret.nonzero()[:, 1]
    log_joint += (MultivariateNormal(mu_ks[labels], cov_ks[labels]).log_prob(Y)).sum()
    log_joint += cat(Pi).log_prob(Z_ret[0])
    log_joint += (cat(A_samples[labels[:-1]]).log_prob(Z_ret[1:])).sum()
    return log_joint
