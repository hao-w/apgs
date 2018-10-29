import matplotlib.pyplot as plt
import torch
from torch import logsumexp
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical

def csmc_hmm(Z_ret, Pi, A, mu_ks, cov_ks, Y, T, D, K, num_particles=1):
    Zs = torch.zeros((num_particles, T, K))
    log_weights = torch.zeros((num_particles, T))
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    log_normalizer = torch.zeros(1).float()

    for t in range(T):
        if t == 0:
            Zs[-1, t] = Z_ret[t]
            label = Z_ret[t].nonzero().item()
            likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
            log_weights[-1, t] = likelihood
            for n in range(num_particles-1):
                sample_z = cat(Pi).sample()
                Zs[n, t] = sample_z
                label = sample_z.nonzero().item()
                likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
                log_weights[n, t] = likelihood

        else:
            reweight = torch.exp(log_weights[:, t-1] - logsumexp(log_weights[:, t-1]))
            ancesters = Categorical(reweight).sample((num_particles-1,))
            Zs[:-1] = Zs[ancesters]
            Zs[-1, t] = Z_ret[t]
            label = Z_ret[t].nonzero().item()
            likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
            log_weights[-1, t] = likelihood
            for n in range(num_particles-1):
                label = torch.mm(Zs[n, t-1].unsqueeze(0), decode_onehot).int().item()
                sample_z = cat(A[label]).sample()
                Zs[n, t] = sample_z
                label = sample_z.nonzero().item()
                likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
                log_weights[n, t] = likelihood
        log_normalizer += logsumexp(log_weights[:, t]) - torch.log(torch.FloatTensor([num_particles]))
    return Zs, log_weights, log_normalizer

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

def log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K):
    log_joint = torch.zeros(1).float()
    decode_onehot = torch.arange(K).float().unsqueeze(-1)
    for t in range(T):
        label = Z_ret[t].nonzero().item()
        likelihood = MultivariateNormal(mu_ks[label], cov_ks[label]).log_prob(Y[t])
        log_joint += likelihood
        if t == 0:
            log_joint += cat(Pi).log_prob(Z_ret[t])
        else:
            label_prev = Z_ret[t-1].nonzero().item()
            log_joint += cat(A_samples[label_prev]).log_prob(Z_ret[t])
    return log_joint
