import matplotlib.pyplot as plt
import time
import torch
from torch import logsumexp
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.categorical import Categorical
from smc import *
from util import *
from data import *


def stepwise_optimal_1(enc, D, K, NUM_EPOCHS, num_particles_rws, num_particles_smc, mcmc_steps):
    """
    At each mcmc step, we each gradient using normalized weights, where the weight is defined as
    p(\eta_t, z_t, Y) * Normalizer_smc_t / \gamma_smc(z_t | \eta_t, Y) * q_\phi(\eta_t | z_{t-1})
    """
    KLs = []
    EUBOs = []
    ESSs = []
    ELBOs = []

    T_min = 50
    T_max = 60
    K = 4
    D = 2
    dt = 5
    Boundary = 30
    noise_ratio = 10.0
    noise_cov = np.array([[1, 0], [0, 1]]) * noise_ratio
    for epoch in range(NUM_EPOCHS):
        alpha_trans_0 = initial_trans_prior(K)
        init_v = init_velocity(dt)
        T = np.random.randint(T_min, T_max)
        STATE, mu_ks, cov_ks, Pi, Y, A_true, Zs_true = generate_seq(T, K, dt, Boundary, init_v, noise_cov)
        posterior = alpha_trans_0 + pairwise(Zs_true, T).sum(0)
        kls = torch.zeros(num_particles_rws)
        Zs_list = []
        ## initialze A from uniform prior, this step does NOT involve gradient
        for l in range(num_particles_rws):
            A_samples = initial_trans(alpha_trans_0, K)
            Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
            Z_ret = resampling_smc(Zs, log_weights)
            Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
            Zs_list.append(Z_ret_pairwise)
        for m in range(mcmc_steps):
            time_start = time.time()
            log_w = torch.zeros(num_particles_rws)
            for l in range(num_particles_rws):
                variational_curr, A_samples = enc(Z_ret_pairwise, alpha_trans_0)
                Zs, log_weights, log_normalizer = csmc_hmm(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
                Z_ret = resampling_smc(Zs, log_weights)
                Z_ret_pairwise = torch.cat((Z_ret[:T-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
                Zs_list[l] = Z_ret_pairwise
                log_w[l] = log_joint(alpha_trans_0, Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K).detach() + log_normalizer - log_joint_smc(Z_ret, Pi, A_samples, mu_ks, cov_ks, Y, T, D, K) - log_q_hmm(variational_curr, A_samples)
                kls[l] = kl_dirichlets(posterior, variational_curr, K)
            weights_rws = torch.exp(log_w - logsumexp(log_w, dim=0)).detach()
            ess = (1. / (weights_rws ** 2 ).sum()).item()
            eubo =  torch.mul(weights_rws, log_w).sum()
            elbo = log_w.mean()
            eubo.backward()
            optimizer.step()
            optimizer.zero_grad()
            EUBOs.append(eubo.item())
            ELBOs.append(elbo.item())
            ESSs.append(ess)
            KLs.append(kls.mean())
            time_end = time.time()
            print('epoch : %d, mcmc : %d, EUBO : %f, ELBO : %f, KL : %f (%ds)' % (epoch, m, eubo, elbo, kls.mean(), time_end - time_start))
    return EUBOs, ELBOs, KLs, ESSs, enc

def flatz(Z, T, K, batch_size):
    return torch.cat((Z[:, :T-1, :].unsqueeze(2), Z[:, 1:, :].unsqueeze(2)), 2).view(batch_size * (T-1), 2*K)

def ag_sis(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Ys, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps, batch_size):
    log_increment_weights = torch.zeros((batch_size, num_particles_rws, mcmc_steps+1))
    log_overall_weights = torch.zeros((batch_size, num_particles_rws, mcmc_steps+1))
    for l in range(num_particles_rws):
        ## As B * K * K
        As = initial_trans(alpha_trans_0, K, batch_size)
        Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
        ## Z B * T * K
        Z = smc_resamplings(Zs, log_weights, batch_size)
        ## z_pairs (B * T-1)-by-(2K)
        Z_pairs = flatz(Z, T, K, batch_size)
        ## the first incremental weight is just log normalizer since As is sampled from prior
        log_increment_weights[:, l, 0] =  log_normalizers
        log_overall_weights[:, l, 0] = log_increment_weights[:, l, 0]
        for m in range(mcmc_steps):
            As_prev = As.clone()
            variational, As = enc(Z_pairs, alpha_trans_0, batch_size)
            log_qs_phi = log_qs(variational, As)
            Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
            Z = smc_resamplings(Zs, log_weights, batch_size)
            Z_pairs = flatz(Z, T, K, batch_size)
            log_qs_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
            log_ps = log_joints(alpha_trans_0, Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size).detach()
            log_increment_weights[:, l, m+1] = log_ps - log_qs_smc + log_normalizers - log_qs_phi
            log_overall_weights[:, l, m+1] = (log_increment_weights[:, l, m+1].clone() + log_overall_weights[:, l, m].clone()).detach()

    log_overall_weights_reshape = log_overall_weights.view(-1, num_particles_rws*(mcmc_steps+1))
    log_increment_weights_reshape = log_increment_weights.view(-1, num_particles_rws*(mcmc_steps+1))
    weights_rws = torch.exp(log_overall_weights_reshape - logsumexp(log_overall_weights_reshape, dim=1).unsqueeze(1)).detach()

    # ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubos =  torch.mul(weights_rws, log_increment_weights_reshape).sum(1)
    elbos =  log_increment_weights.sum(-1).mean(1)
    return eubos.mean(), elbos.mean(), variational

def oneshot(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Ys, T, D, K, num_particles_rws, num_particles_smc, mcmc_steps, batch_size):
    log_increment_weights = torch.zeros((batch_size, num_particles_rws, mcmc_steps+1))
    log_overall_weights = torch.zeros((batch_size, num_particles_rws, mcmc_steps+1))
    for l in range(num_particles_rws):
        ## As B * K * K
        As = initial_trans(alpha_trans_0, K, batch_size)
        Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
        ## Z B * T * K
        Z = smc_resamplings(Zs, log_weights, batch_size)
        ## z_pairs (B * T-1)-by-(2K)
        Z_pairs = flatz(Z, T, K, batch_size)
        ## the first incremental weight is just log normalizer since As is sampled from prior
        log_increment_weights[:, l, 0] =  log_normalizers
        log_overall_weights[:, l, 0] = log_increment_weights[:, l, 0]
        for m in range(mcmc_steps):
            As_prev = As.clone()
            variational, As = enc(Z_pairs, alpha_trans_0, batch_size)
            log_qs_phi = log_qs(variational, As)
            Zs, log_weights, log_normalizers = smc_hmm_batch(Pi, As, mu_ks, cov_ks, Ys, T, D, K, num_particles_smc, batch_size)
            Z = smc_resamplings(Zs, log_weights, batch_size)
            Z_pairs = flatz(Z, T, K, batch_size)
            log_qs_smc = smc_log_joints(Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size)
            log_ps = log_joints(alpha_trans_0, Z, Pi, As, mu_ks, cov_ks, Ys, T, D, K, batch_size).detach()
            log_increment_weights[:, l, m+1] = log_ps - log_qs_smc + log_normalizers - log_qs_phi
            log_overall_weights[:, l, m+1] = (log_increment_weights[:, l, m+1].clone() + log_overall_weights[:, l, m].clone()).detach()

    log_overall_weights_reshape = log_overall_weights.view(-1, num_particles_rws*(mcmc_steps+1))
    log_increment_weights_reshape = log_increment_weights.view(-1, num_particles_rws*(mcmc_steps+1))
    weights_rws = torch.exp(log_overall_weights_reshape - logsumexp(log_overall_weights_reshape, dim=1).unsqueeze(1)).detach()

    # ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubos =  torch.mul(weights_rws, log_increment_weights_reshape).sum(1)
    elbos =  log_increment_weights.sum(-1).mean(1)
    return eubos.mean(), elbos.mean(), variational
