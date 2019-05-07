import torch
import time
from utils import *

def train_cfz_reparam(Eubo, enc_eta, gibbs_z, optimizer, Data, K, num_epochs, mcmc_size, sample_size, batch_size, PATH, CUDA, device):
    NUM_SEQS, N, D = Data.shape
    num_batches = int((NUM_SEQS / batch_size))

    flog = open('../results/log-' + PATH + '.txt', 'w+')
    flog.write('SymKL\tEUBO\tELBO\tESS\tKLs_eta_ex\tKLs_eta_in\tKLs_z_ex\tKLs_z_in\n')
    flog.close()

    for epoch in range(num_epochs):
        time_start = time.time()
        indices = torch.randperm(NUM_SEQS)
        EUBO = 0.0
        ELBO = 0.0
        ESS = 0.0
        SymKL = 0.0
        KL_eta_ex = 0.0
        KL_eta_in = 0.0
        KL_z_ex = 0.0
        KL_z_in = 0.0
        for step in range(num_batches):
            optimizer.zero_grad()
            batch_indices = indices[step*batch_size : (step+1)*batch_size]
            obs = Data[batch_indices]
            obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
            if CUDA:
                obs =obs.cuda().to(device)
            symkls, eubos, elbos, esss, q_eta, p_eta, q_z, p_z, q_nu, pr_nu = Eubo(enc_eta, gibbs_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device)
            kl_eta_ex, kl_eta_in, kl_z_ex, kl_z_in = kl_train(q_eta, p_eta, q_z, p_z, q_nu, pr_nu, obs, K)
            ## gradient step
            eubos.sum().backward()
            optimizer.step()
            SymKL += symkls.sum().item()
            EUBO += eubos.sum().item()
            ELBO += elbos.sum().item()
            ESS += esss.mean().item()
            KL_eta_ex += kl_eta_ex.item()
            KL_eta_in += kl_eta_in.item()
            KL_z_ex += kl_z_ex.item()
            KL_z_in += kl_z_in.item()

        flog = open('../results/log-' + PATH + '.txt', 'a+')
        print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                % (SymKL/num_batches, EUBO/num_batches, ELBO/num_batches, ESS/num_batches, KL_eta_ex/num_batches, KL_eta_in/num_batches, KL_z_ex/num_batches, KL_z_in/num_batches), file=flog)
        flog.close()
        time_end = time.time()
        print('epoch=%d, symKL=%.3f, EUBO=%.3f, ELBO=%.3f, ESS=%.3f (%ds)'
                % (epoch, SymKL/num_batches, EUBO/num_batches, ELBO/num_batches, ESS/num_batches, time_end - time_start))

def test(Eubo, enc_eta, enc_z, Data, K, mcmc_size, sample_size, batch_size, CUDA, device):
    NUM_SEQS, N, D = Data.shape
    indices = torch.randperm(NUM_SEQS)
    batch_indices = indices[0*batch_size : (0+1)*batch_size]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    _, _, _, q_eta, p_eta, q_z, p_z, _, _ = Eubo(enc_eta, enc_z, obs, N, K, D, mcmc_size, sample_size, batch_size, device)
    return obs, q_eta, q_z
