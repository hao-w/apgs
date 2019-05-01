import torch
import time
from eubo import *
from utils import True_Log_likelihood, shuffler

def train(Eubo, enc_mu, enc_z, optimizer, Data, radius, K, num_epochs, mcmc_size, sample_size, batch_size, PATH, CUDA, device):
    NUM_SEQS, N, D = Data.shape
    num_batches = int((NUM_SEQS / batch_size))
    EUBOs = []
    ELBOs = []
    ESSs = []
    ## fixed radius
    obs_rad = torch.ones(1) * radius
    if CUDA:
        obs_rad = obs_rad.cuda().to(device)
    flog = open('../results/log-' + PATH + '.txt', 'w+')
    flog.write('EUBO\tELBO\tESS\n')
    flog.close()
    for epoch in range(num_epochs):
        time_start = time.time()
        indices = torch.randperm(NUM_SEQS)
        EUBO = 0.0
        ELBO = 0.0
        ESS = 0.0
        for step in range(num_batches):
            optimizer.zero_grad()
            batch_indices = indices[step*batch_size : (step+1)*batch_size]
            obs = Data[batch_indices]
            obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
            if CUDA:
                obs = obs.cuda().to(device)
            eubos, elbos, esss = Eubo(enc_mu, enc_z, obs, obs_rad, N, K, D, mcmc_size, sample_size, batch_size, device, noise_sigma=0.05)
            eubos.mean().backward()
            optimizer.step()
            EUBO += eubos[-1].item()
            ELBO += elbos[-1].item()
            ESS += esss[-1].item()
        EUBOs.append(EUBO / num_batches)
        ELBOs.append(ELBO / num_batches)
        ESSs.append(ESS / num_batches)
        flog = open('../results/log-' + PATH + '.txt', 'a+')
        print('%.3f\t%.3f\t%.3f'
                % (EUBO/num_batches, ELBO/num_batches, ESS/num_batches), file=flog)
        flog.close()
        time_end = time.time()
        print('epoch=%d, EUBO=%.3f, ELBO=%.3f, ESS=%.3f (%ds)'
                % (epoch, EUBO/num_batches, ELBO/num_batches, ESS/num_batches,
                   time_end - time_start))

def test(Eubo, enc_mu, enc_z, Data, radius, K, mcmc_size, sample_size, batch_size, CUDA, device):
    NUM_SEQS, N, D = Data.shape
    num_batches = int((NUM_SEQS / batch_size))
    ## fixed radius
    obs_rad = torch.ones(1) * radius
    if CUDA:
        obs_rad = obs_rad.cuda().to(device)
    indices = torch.randperm(NUM_SEQS)
    batch_indices = indices[0*batch_size : (0+1)*batch_size]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
    if CUDA:
        obs = obs.cuda().to(device)
    _, _, _, q_mu, q_z = Eubo(enc_mu, enc_z, obs, obs_rad, N, K, D, mcmc_size, sample_size, batch_size, device, noise_sigma=0.05)
    return obs, q_mu, q_z
