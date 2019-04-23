import torch
import torch.nn as nn
from utils import *
from kls import *
from NG_nats import *
import sys
sys.path.append('/home/hao/Research/probtorch/')
import probtorch
from torch.distributions.categorical import Categorical

def Eubo_ag_sis_initz_adapt_idw(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, SAMPLE_DIM=0, BATCH_DIM=1):
    """
    initialize z
    adaptive resampling after each step
    independently IW for each variable
    """
    eubos = torch.zeros(mcmc_size).cuda()
    elbos = torch.zeros(mcmc_size).cuda()
    esss = torch.zeros(mcmc_size).cuda()

    for m in range(mcmc_size):
        if m == 0:
            p_init_z = cat(enc_z.prior_pi)
            states = p_init_z.sample((sample_size, batch_size, N,))
            log_p_z = p_init_z.log_prob(states).sum(-1)## S * B * N
            log_q_z = p_init_z.log_prob(states).sum(-1)
        else:
            ## adaptive resampling
            obs_mu, obs_sigma = resample_eta(obs_mu, obs_sigma, weights)
            ## update z -- cluster assignments
            q_z, p_z = enc_z(obs, obs_sigma, obs_mu, sample_size, batch_size)
            log_p_z = p_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_q_z = q_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            states = q_z['zs'].value ## S * B * N * K
        ## update tau and mu -- global variables
        local_vars = torch.cat((obs, states), -1)
        q_eta, p_eta, q_nu = enc_eta(local_vars)
        log_p_eta = p_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
        log_q_eta = q_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)

        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        obs_sigma = 1. / obs_tau.sqrt()
        ##
        log_obs = Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=False)
        log_weights = log_obs.sum(-1) + log_p_eta + log_p_z - log_q_eta - log_q_z
        weights = F.softmax(log_weights, 0).detach()
        eubos[m] = (weights * log_weights).sum(0).mean()
        elbos[m] = log_weights.mean()
        esss[m] = (1. / (weights**2).sum(0)).mean()

    ## KLs for mu and sigma based on Normal-Gamma prior
    q_mu = q_eta['means'].dist.loc
    q_alpha = q_eta['precisions'].dist.concentration
    q_beta = q_eta['precisions'].dist.rate
    q_logits = q_z['zs'].dist.probs.log()
    stat1, stat2, stat3 = data_to_stats(obs, states, K, D)
    post_alpha, post_beta, post_mu, post_nu = Post_mu_tau(stat1, stat2, stat3, enc_eta.prior_alpha, enc_eta.prior_beta, enc_eta.prior_mu, enc_eta.prior_nu, D)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    ## KLs for cluster assignments
    post_logits = Post_z(obs, obs_sigma, obs_mu, N, K)
    kl_z_ex, kl_z_in = kls_cats(q_logits, post_logits)

    return eubos.mean(), elbos.mean(), esss.mean(), kl_eta_ex.sum(-1).mean(), kl_eta_in.sum(-1).mean(), kl_z_ex.sum(-1).mean(), kl_z_in.sum(-1).mean()

def Eubo_ag_sis_initz(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, SAMPLE_DIM=0, BATCH_DIM=1):
    """
    initialize z
    """
    eubos = torch.zeros(mcmc_size).cuda()
    elbos = torch.zeros(mcmc_size).cuda()
    esss = torch.zeros(mcmc_size).cuda()

    for m in range(mcmc_size):
        if m == 0:
            p_init_z = cat(enc_z.prior_pi)
            states = p_init_z.sample((sample_size, batch_size, N,))
            log_p_z = p_init_z.log_prob(states).sum(-1)## S * B * N
            log_q_z = p_init_z.log_prob(states).sum(-1)
        else:
            ## update z -- cluster assignments
            q_z, p_z = enc_z(obs, obs_sigma, obs_mu, sample_size, batch_size)
            log_p_z = p_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_q_z = q_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            states = q_z['zs'].value ## S * B * N * K
        ## update tau and mu -- global variables
        local_vars = torch.cat((obs, states), -1)
        q_eta, p_eta, q_nu = enc_eta(local_vars)
        log_p_eta = p_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
        log_q_eta = q_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)

        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        obs_sigma = 1. / obs_tau.sqrt()
        ##
        log_obs = Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=False)
        log_weights = log_obs.sum(-1) + log_p_eta + log_p_z - log_q_eta - log_q_z
        weights = F.softmax(log_weights, 0).detach()
        eubos[m] = (weights * log_weights).sum(0).mean()
        elbos[m] = log_weights.mean()
        esss[m] = (1. / (weights**2).sum(0)).mean()

    ## KLs for mu and sigma based on Normal-Gamma prior
    q_mu = q_eta['means'].dist.loc
    q_alpha = q_eta['precisions'].dist.concentration
    q_beta = q_eta['precisions'].dist.rate
    q_logits = q_z['zs'].dist.probs.log()
    stat1, stat2, stat3 = data_to_stats(obs, states, K, D)
    post_alpha, post_beta, post_mu, post_nu = Post_mu_tau(stat1, stat2, stat3, enc_eta.prior_alpha, enc_eta.prior_beta, enc_eta.prior_mu, enc_eta.prior_nu, D)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    ## KLs for cluster assignments
    post_logits = Post_z(obs, obs_sigma, obs_mu, N, K)
    kl_z_ex, kl_z_in = kls_cats(q_logits, post_logits)

    return eubos.mean(), elbos.mean(), esss.mean(), kl_eta_ex.sum(-1).mean(), kl_eta_in.sum(-1).mean(), kl_z_ex.sum(-1).mean(), kl_z_in.sum(-1).mean()


def resample_eta(obs_mu, obs_sigma, weights):
    """
    weights is S * B
    """
    S, B, K, D = obs_mu.shape
    ancesters = Categorical(weights.transpose(0,1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, D) ## S * B * K * D
    obs_mu_r = torch.gather(obs_mu, 0, ancesters)
    obs_sigma_r = torch.gather(obs_sigma, 0, ancesters)
    return obs_mu_r, obs_sigma_r

def Eubo_ag_sis_initz_adapt(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, SAMPLE_DIM=0, BATCH_DIM=1):
    """
    initialize z
    adaptive resampling after each step
    """
    eubos = torch.zeros(mcmc_size).cuda()
    elbos = torch.zeros(mcmc_size).cuda()
    esss = torch.zeros(mcmc_size).cuda()

    for m in range(mcmc_size):
        if m == 0:
            p_init_z = cat(enc_z.prior_pi)
            states = p_init_z.sample((sample_size, batch_size, N,))
            log_p_z = p_init_z.log_prob(states).sum(-1)## S * B * N
            log_q_z = p_init_z.log_prob(states).sum(-1)
        else:
            ## adaptive resampling
            obs_mu, obs_sigma = resample_eta(obs_mu, obs_sigma, weights)
            ## update z -- cluster assignments
            q_z, p_z = enc_z(obs, obs_sigma, obs_mu, sample_size, batch_size)
            log_p_z = p_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            log_q_z = q_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
            states = q_z['zs'].value ## S * B * N * K
        ## update tau and mu -- global variables
        local_vars = torch.cat((obs, states), -1)
        q_eta, p_eta, q_nu = enc_eta(local_vars)
        log_p_eta = p_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
        log_q_eta = q_eta.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)

        obs_mu = q_eta['means'].value
        obs_tau = q_eta['precisions'].value
        obs_sigma = 1. / obs_tau.sqrt()
        ##
        log_obs = Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=False)
        log_weights = log_obs.sum(-1) + log_p_eta + log_p_z - log_q_eta - log_q_z
        weights = F.softmax(log_weights, 0).detach()
        eubos[m] = (weights * log_weights).sum(0).mean()
        elbos[m] = log_weights.mean()
        esss[m] = (1. / (weights**2).sum(0)).mean()

    ## KLs for mu and sigma based on Normal-Gamma prior
    q_mu = q_eta['means'].dist.loc
    q_alpha = q_eta['precisions'].dist.concentration
    q_beta = q_eta['precisions'].dist.rate
    q_logits = q_z['zs'].dist.probs.log()
    stat1, stat2, stat3 = data_to_stats(obs, states, K, D)
    post_alpha, post_beta, post_mu, post_nu = Post_mu_tau(stat1, stat2, stat3, enc_eta.prior_alpha, enc_eta.prior_beta, enc_eta.prior_mu, enc_eta.prior_nu, D)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    ## KLs for cluster assignments
    post_logits = Post_z(obs, obs_sigma, obs_mu, N, K)
    kl_z_ex, kl_z_in = kls_cats(q_logits, post_logits)

    return eubos.mean(), elbos.mean(), esss.mean(), kl_eta_ex.sum(-1).mean(), kl_eta_in.sum(-1).mean(), kl_z_ex.sum(-1).mean(), kl_z_in.sum(-1).mean()

def Eubo_ag_sis_initeta(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, SAMPLE_DIM=0, BATCH_DIM=1):
    """
    initialize eta
    incremental weight doesn't involve backward transition
    """
    eubos = torch.zeros(mcmc_size).cuda()
    elbos = torch.zeros(mcmc_size).cuda()
    esss = torch.zeros(mcmc_size).cuda()

    for m in range(mcmc_size):
        if m == 0:
            prior_mu_expand = enc_eta.prior_mu.unsqueeze(0).unsqueeze(0).repeat(sample_size, batch_size, 1, 1)
            p_init_tau = Gamma(enc_eta.prior_alpha, enc_eta.prior_alpha)
            obs_tau = p_init_tau.sample((sample_size, batch_size,))
            p_init_mu = Normal(prior_mu_expand, 1. / (enc_eta.prior_nu * obs_tau).sqrt())
            obs_mu = p_init_mu.sample()
            log_p_eta = p_init_tau.log_prob(obs_tau).sum(-1).sum(-1) + p_init_mu.log_prob(obs_mu).sum(-1).sum(-1)## S * B
            log_q_eta = log_p_eta
            obs_sigma = 1. / obs_tau.sqrt()
        else:
            ## update tau and mu -- global variables
            local_vars = torch.cat((obs, states), -1)
            q_eta, p_eta, q_nu = enc_eta(local_vars)
            log_p_eta = p_eta.log_joint(sample_dims=0, batch_dim=1)
            log_q_eta = q_eta.log_joint(sample_dims=0, batch_dim=1)
            ## for individual importance weight, S * B * K
            obs_mu = q_eta['means'].value
            obs_tau = q_eta['precisions'].value
            obs_sigma = 1. / obs_tau.sqrt()
        ## update z -- cluster assignments
        q_z, p_z = enc_z(obs, obs_sigma, obs_mu, sample_size, batch_size)
        log_p_z = p_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
        log_q_z = q_z.log_joint(sample_dims=SAMPLE_DIM, batch_dim=BATCH_DIM)
        states = q_z['zs'].value ## S * B * N * K
        log_obs = Log_likelihood(obs, states, obs_mu, obs_sigma, K, D, cluster_flag=False)
        log_weights = log_obs.sum(-1) + log_p_eta + log_p_z - log_q_eta - log_q_z
        weights = F.softmax(log_weights, 0).detach()
        eubos[m] = (weights * log_weights).sum(0).mean()
        elbos[m] = log_weights.mean()
        esss[m] = (1. / (weights**2).sum(0)).mean()
    ## KLs for mu and sigma based on Normal-Gamma prior
    q_mu = q_eta['means'].dist.loc
    q_alpha = q_eta['precisions'].dist.concentration
    q_beta = q_eta['precisions'].dist.rate
    q_logits = q_z['zs'].dist.probs.log()
    stat1, stat2, stat3 = data_to_stats(obs, states, K, D)
    post_alpha, post_beta, post_mu, post_nu = Post_mu_tau(stat1, stat2, stat3, enc_eta.prior_alpha, enc_eta.prior_beta, enc_eta.prior_mu, enc_eta.prior_nu, D)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    ## KLs for cluster assignments
    post_logits = Post_z(obs, obs_sigma, obs_mu, N, K)
    kl_z_ex, kl_z_in = kls_cats(q_logits, post_logits)

    return eubos.mean(), elbos.mean(), esss.mean(), kl_eta_ex.sum(-1).mean(), kl_eta_in.sum(-1).mean(), kl_z_ex.sum(-1).mean(), kl_z_in.sum(-1).mean()
