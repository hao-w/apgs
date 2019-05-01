import torch
import time
from utils import *
from normal_gamma_kls import *
from normal_gamma_conjugacy import *

def train(Eubo, enc_eta, enc_z, optimizer, Data, K, NUM_EPOCHS, MCMC_SIZE, SAMPLE_SIZE, BATCH_SIZE, PATH, CUDA, device):
    EUBOs = []
    ELBOs = []
    ESSs = []
    NUM_SEQS, N, D = Data.shape
    NUM_BATCHES = int((NUM_SEQS / BATCH_SIZE))

    flog = open('../results/log-' + PATH + '.txt', 'w+')
    flog.write('EUBO\tELBO\tESS\tKLs_eta_ex\tKLs_eta_in\tKLs_z_ex\tKLs_z_in\n')
    flog.close()

    for epoch in range(NUM_EPOCHS):
        time_start = time.time()
        indices = torch.randperm(NUM_SEQS)
        EUBO = 0.0
        ELBO = 0.0
        ESS = 0.0
        KL_eta_ex = 0.0
        KL_eta_in = 0.0
        KL_z_ex = 0.0
        KL_z_in = 0.0
        for step in range(NUM_BATCHES):
            optimizer.zero_grad()
            batch_indices = indices[step*BATCH_SIZE : (step+1)*BATCH_SIZE]
            obs = Data[batch_indices]
            obs = shuffler(obs).repeat(SAMPLE_SIZE, 1, 1, 1)
            if CUDA:
                obs =obs.cuda().to(device)
            eubo, elbo, ess, q_eta, p_eta, q_z, p_z, q_nu, pr_nu = Eubo(enc_eta, enc_z, obs, N, K, D, MCMC_SIZE, SAMPLE_SIZE, BATCH_SIZE, device)
            kl_eta_ex, kl_eta_in, kl_z_ex, kl_z_in = kl_train(q_eta, p_eta, q_z, p_z, q_nu, pr_nu, obs, K)
            ## gradient step
            eubo.backward()
            optimizer.step()
            EUBO += eubo.item()
            ELBO += elbo.item()
            ESS += ess.item()
            KL_eta_ex += kl_eta_ex.item()
            KL_eta_in += kl_eta_in.item()
            KL_z_ex += kl_z_ex.item()
            KL_z_in += kl_z_in.item()
        EUBOs.append(EUBO / NUM_BATCHES)
        ELBOs.append(ELBO / NUM_BATCHES)
        ESSs.append(ESS / NUM_BATCHES)
        flog = open('../results/log-' + PATH + '.txt', 'a+')
        print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                % (EUBO/NUM_BATCHES, ELBO/NUM_BATCHES, ESS/NUM_BATCHES, KL_eta_ex/NUM_BATCHES, KL_eta_in/NUM_BATCHES, KL_z_ex/NUM_BATCHES, KL_z_in/NUM_BATCHES), file=flog)
        flog.close()
        time_end = time.time()
        print('epoch=%d, EUBO=%.3f, ELBO=%.3f, ESS=%.3f (%ds)'
                % (epoch, EUBO/NUM_BATCHES, ELBO/NUM_BATCHES, ESS/NUM_BATCHES, time_end - time_start))

def kl_train(q_eta, p_eta, q_z, p_z, q_nu, pr_nu, obs, K):
    _, _, N, D = obs.shape
    ## KLs for mu and sigma based on Normal-Gamma prior
    q_alpha = q_eta['precisions'].dist.concentration
    q_beta = q_eta['precisions'].dist.rate
    q_mu = q_eta['means'].dist.loc
    q_pi = q_z['zs'].dist.probs

    pr_alpha = p_eta['precisions'].dist.concentration
    pr_beta = p_eta['precisions'].dist.rate
    pr_mu = p_eta['means'].dist.loc
    pr_pi = p_z['zs'].dist.probs

    states = q_z['zs'].value
    obs_mu = q_eta['means'].value
    obs_sigma = 1. / q_eta['precisions'].value.sqrt()

    post_alpha, post_beta, post_mu, post_nu = Post_eta(obs, states, pr_alpha, pr_beta, pr_mu, pr_nu, K, D)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    ## KLs for cluster assignments
    post_logits = Post_z(obs, obs_sigma, obs_mu, pr_pi, N, K)
    kl_z_ex, kl_z_in = kls_cats(q_pi.log(), post_logits)
    return kl_eta_ex.sum(-1).mean(), kl_eta_in.sum(-1).mean(), kl_z_ex.sum(-1).mean(), kl_z_in.sum(-1).mean()

def test(Eubo, enc_eta, enc_z, Data, K, MCMC_SIZE, SAMPLE_SIZE, BATCH_SIZE, CUDA, device):
    NUM_SEQS, N, D = Data.shape
    indices = torch.randperm(NUM_SEQS)
    batch_indices = indices[0*batch_size : (0+1)*batch_size]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    _, _, _, q_eta, p_eta, q_z, p_z, _, _ = Eubo(enc_eta, enc_z, obs, N, K, D, MCMC_SIZE, SAMPLE_SIZE, BATCH_SIZE, device)
    return q_eta, q_z
