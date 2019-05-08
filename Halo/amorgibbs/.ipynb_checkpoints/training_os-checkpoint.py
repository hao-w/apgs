import torch
import time
from eubo import *
from utils import True_Log_likelihood, shuffler

def train_mu(Eubo, oneshot_mu, enc_mu, enc_z, optimizer, Data, obs_rad, noise_sigma, K, num_epochs, mcmc_size, sample_size, batch_size, PATH, CUDA, device, RESAMPLE, DETACH=True):
    NUM_SEQS, N, D = Data.shape
    num_batches = int((NUM_SEQS / batch_size))

    flog = open('../results/log-' + PATH + '.txt', 'w+')
    flog.write('EUBO\tELBO\tESS\n')
    flog.close()
    for epoch in range(num_epochs):
        time_start = time.time()
        indices = torch.randperm(NUM_SEQS)
        EUBO = 0.0
        ELBO = 0.0
        ESS = 0.0
        SymKL = 0.0
        for step in range(num_batches):
            optimizer.zero_grad()
            batch_indices = indices[step*batch_size : (step+1)*batch_size]
            obs = Data[batch_indices]
            obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
            if CUDA:
                obs = obs.cuda().to(device)
            symkls, eubos, elbos, esss, _, _, _, _ = Eubo(oneshot_mu, enc_mu, enc_z, obs, obs_rad, noise_sigma, N, K, D, mcmc_size, sample_size, batch_size, device, RESAMPLE, DETACH=DETACH)
            symkls.mean().backward()
            optimizer.step()
            SymKL += symkls.sum().item()
            EUBO += eubos.sum().item()
            ELBO += elbos.sum().item()
            ESS += esss.mean().item()
        flog = open('../results/log-' + PATH + '.txt', 'a+')
        print('%.3f\t%.3f\t%.3f\t%.3f'
                % (SymKL/num_batches, EUBO/num_batches, ELBO/num_batches, ESS/num_batches), file=flog)
        flog.close()
        time_end = time.time()
        print('epoch=%d, SymKL=%.3f, EUBO=%.3f, ELBO=%.3f, ESS=%.3f (%ds)'
                % (epoch, SymKL/num_batches, EUBO/num_batches, ELBO/num_batches, ESS/num_batches,
                   time_end - time_start))

def test(Eubo, oneshot_mu, enc_mu, enc_z, Data, obs_rad, noise_sigma, K, mcmc_size, sample_size, batch_size, CUDA, device, RESAMPLE, DETACH):
    NUM_SEQS, N, D = Data.shape
    num_batches = int((NUM_SEQS / batch_size))
    indices = torch.randperm(NUM_SEQS)
    batch_indices = indices[0*batch_size : (0+1)*batch_size]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
    if CUDA:
        obs = obs.cuda().to(device)
    symkls_test, eubos_test, elbos_test, _, q_mu, _, q_z, _ = Eubo(oneshot_mu, enc_mu, enc_z, obs, obs_rad, noise_sigma, N, K, D, mcmc_size, sample_size, batch_size, device, RESAMPLE, DETACH)
    return obs, q_mu, q_z, symkls_test, eubos_test, elbos_test