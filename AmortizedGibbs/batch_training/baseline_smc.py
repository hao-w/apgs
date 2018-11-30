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
    
def oneshot_sampling(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles, num_particles_smc):
    log_weights_is = torch.zeros(num_particles)
    log_p_joints = torch.zeros(num_particles)
    kls = torch.zeros(num_particles)
    for l in range(num_particles):
        Y_pairwise = torch.cat((Y[:-1].unsqueeze(0), Y[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*D)
        variational, A_sample = enc(Y_pairwise, alpha_trans_0)
        log_q_mlp = log_q_hmm(variational, A_sample)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_sample, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_q_smc = log_joint_smc(Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K)
        log_p_joints = log_joint(alpha_trans_0, Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K).detach()
        log_weights_is[l] = log_p_joints - log_q_smc - log_q_mlp + log_normalizer

        posterior = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(posterior, variational, K)

    log_weights_is_norm = log_weights_is - logsumexp(log_weights_is, dim=0)
    weights_is = torch.exp(log_weights_is_norm).detach()
    kl =  kls.mean()
    ess = (1. / (weights_is ** 2 ).sum()).item()
    eubo =  torch.mul(weights_is, log_weights_is).sum()
    elbo = log_weights_is.mean()
    return eubo, kl, ess, variational, elbo

def twoshots_sampling(enc_trans, enc_disp, alpha_trans_0, Pi, mu_ks, cov_ks, Z_true, Y, T, D, K, num_particles, num_particles_smc):
    log_weights_rws = torch.zeros(num_particles)
    log_p_joints = torch.zeros(num_particles)
    kls = torch.zeros(num_particles)
    for l in range(num_particles):
        Y_pairwise = torch.cat((Y[:-1].unsqueeze(0), Y[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*D)
        latents_dirs_disp, A_sample_disp = enc_disp(Y_pairwise)
        Zs, log_weights, log_normalizer = smc_hmm(Pi, A_sample_disp, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)
        log_p_joints = log_joint(alpha_trans_0, Z_ret, Pi, A_sample_disp, mu_ks, cov_ks, Y, T, D, K).detach()
        log_q_smc = log_joint_smc(Z_ret, Pi, A_sample_disp, mu_ks, cov_ks, Y, T, D, K)
        log_q_enc_disp = log_q_hmm(latents_dirs_disp, A_sample_disp)
        log_weights_rws[l] = log_p_joints - log_q_smc - log_q_enc_disp + log_normalizer

        Z_pairwise = torch.cat((Z_ret[:-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
        latents_dirs_trans, A_sample_trans = enc_trans(Z_pairwise)
        log_p_joints_target = log_joint(alpha_trans_0, Z_ret, Pi, A_sample_trans, mu_ks, cov_ks, Y, T, D, K).detach()
        log_q_forward = log_q_hmm(latents_dirs_trans, A_sample_trans)
        log_q_reverse = log_q_hmm(latents_dirs_trans, A_sample_disp)
        log_weights_rws[l] = log_weights_rws[l] + log_p_joints_targetalpha_trans_0, latents_dirs, Z_ret, T, K - log_p_joints + log_q_reverse - log_q_forward

        Zs, log_weights, log_normalizer = csmc_hmm(Z_ret, Pi, A_sample_trans, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
        Z_ret = resampling_smc(Zs, log_weights)

        kls[l] = kl_dirichlets(alpha_trans_0, latents_dirs_trans, Z_ret, T, K)

    weights_rws = torch.exp(log_weights_rws - logsumexp(log_weights_rws, dim=0)).detach()
    kl =  kls.mean()
    ess = (1. / (weights_rws ** 2 ).sum()).item()
    eubo =  torch.mul(weights_rws, log_weights_rws).sum()
    elbo = log_weights_rws.mean()
    return eubo, kl, ess, latents_dirs_trans, elbo

def oneshot_sharp_prior(enc, alpha_trans_0, Pi, mu_ks, cov_ks, Y, T, D, K, num_particles, num_particles_smc):
    log_weights_is = torch.zeros(num_particles)
    log_p_joints = torch.zeros(num_particles)
    kls = torch.zeros(num_particles)

    A_samples = initial_trans(alpha_trans_0, K)

    Zs, log_weights, log_normalizer = smc_hmm(Pi, A_samples, mu_ks, cov_ks, Y, T, D, K, num_particles_smc)
    Z_ret = resampling_smc(Zs, log_weights)
    Z_ret_pairs = torch.cat((Z_ret[:-1].unsqueeze(0), Z_ret[1:].unsqueeze(0)), 0).transpose(0, 1).contiguous().view(T-1, 2*K)
    for l in range(num_particles):
        variational, A_sample = enc(Z_ret_pairs, alpha_trans_0)
        log_q_mlp = log_q_hmm(variational, A_sample)
        log_q_smc = log_joint_smc(Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K)
        log_p_joints = log_joint(alpha_trans_0, Z_ret, Pi, A_sample, mu_ks, cov_ks, Y, T, D, K).detach()
        log_weights_is[l] = log_p_joints - log_q_smc - log_q_mlp + log_normalizer
        posterior = alpha_trans_0 + pairwise(Z_ret, T).sum(0)
        kls[l] = kl_dirichlets(posterior, variational, K)

    weights_is = torch.exp(log_weights_is - logsumexp(log_weights_is, dim=0)).detach()
    kl =  kls.mean()
    ess = (1. / (weights_is ** 2 ).sum()).item()
    eubo =  torch.mul(weights_is, log_weights_is).sum()
    elbo = log_weights_is.mean()
    return eubo, kl, ess, variational, elbo
