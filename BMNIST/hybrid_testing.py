import sys
sys.path.append('../')
import torch
import numpy as np
from random import shuffle
from resample import Resampler
from hmc_objective import HMC
from hybrid_objective import hybrid_objective
import time
import numpy as np
import os

def sample_data_uniform(DATA_PATHS, data_ptr):
    """
    ==========
    randomly select one bmnist file from the list of data paths
    ==========
    """
    shuffle(DATA_PATHS)
    data = torch.from_numpy(np.load(DATA_PATHS[0])).float() ## load testing datasets
    return data[data_ptr]

def test_hybrid(num_runs, model, flags, AT, apg_sweeps, frames, mnist_mean_path, K, D, z_what_dim, sample_size, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, CUDA, DEVICE):
    DENSITIES = {'apg' : [], 'hmc' : [], 'bpg' : []}
    mnist_mean = torch.from_numpy(np.load(mnist_mean_path)).float().repeat(K, 1, 1).unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
    frames = frames.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
    if CUDA:
        with torch.cuda.device(DEVICE):
            frames = frames.cuda()
            mnist_mean = mnist_mean.cuda()
    S, B, T, _, _ = frames.shape
    hmc_sampler = HMC(model=model,
                      AT=AT,
                      S=S,
                      B=B,
                      T=T,
                      K=K,
                      D=D,
                      z_what_dim=z_what_dim,
                      hmc_num_steps=hmc_num_steps,
                      step_size_what=leapfrog_step_size,
                      step_size_where=leapfrog_step_size,
                      leapfrog_num_steps=leapfrog_num_steps,
                      CUDA=CUDA,
                      DEVICE=DEVICE)

    resampler = Resampler(strategy='systematic',
                          sample_size=sample_size,
                          CUDA=CUDA,
                          DEVICE=DEVICE)
    for r in range(num_runs):
        time_start = time.time()
        densities = hybrid_objective(model=model,
                                        flags=flags,
                                        AT=AT,
                                        hmc=hmc_sampler,
                                        resampler=resampler,
                                        apg_sweeps=apg_sweeps,
                                        frames=frames,
                                        mnist_mean=mnist_mean,
                                        K=K)
        if flags['apg']:
            np.save('log_joint_apg_run=%d' % (r+1), densities['apg'].cpu().squeeze(-1).data.numpy())
        if flags['hmc']:
            np.save('log_joint_hmc_run=%d' % (r+1), densities['hmc'].cpu().squeeze(-1).data.numpy())
        if flags['bpg']:
            np.save('log_joint_bpg_run=%d' % (r+1), densities['bpg'].cpu().squeeze(-1).data.numpy())
        time_end = time.time()
        print('Run=%d/%d completed in %ds' % (r+1, num_runs, time_end - time_start))
    for r in range(num_runs):
        if flags['apg']:
            DENSITIES['apg'].append(np.load('log_joint_apg_run=%d.npy' % (r+1)).mean(1)[None, :])
            os.remove('log_joint_apg_run=%d.npy' % (r+1))
        if flags['hmc']:
            DENSITIES['hmc'].append(np.load('log_joint_hmc_run=%d.npy' % (r+1)).mean(1)[None, :])
            os.remove('log_joint_hmc_run=%d.npy' % (r+1))
        if flags['bpg']:
            DENSITIES['bpg'].append(np.load('log_joint_bpg_run=%d.npy' % (r+1)).mean(1)[None, :])
            os.remove('log_joint_bpg_run=%d.npy' % (r+1))
    print('Merged individual results..')
    np.save('log_joint_apg', np.concatenate(DENSITIES['apg'], 0))
    np.save('log_joint_hmc', np.concatenate(DENSITIES['hmc'], 0))
    np.save('log_joint_bpg', np.concatenate(DENSITIES['bpg'], 0))


def test_hybrid_all(model, flags, AT, apg_sweeps, data_paths, mnist_mean_path, T, K, D, z_what_dim, batch_size, sample_size, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, CUDA, DEVICE):
    metrics = {'apg' : [], 'hmc' : [], "bpg" : []}
    mnist_mean = torch.from_numpy(np.load(mnist_mean_path)).float().unsqueeze(0).repeat(K, 1, 1).unsqueeze(0).unsqueeze(0).repeat(sample_size, batch_size, 1, 1, 1)
    resampler = Resampler(strategy='systematic',
                          sample_size=sample_size,
                          CUDA=CUDA,
                          DEVICE=DEVICE)
    hmc_sampler = HMC(model=model,
                      AT=AT,
                      S=sample_size,
                      B=batch_size,
                      T=T,
                      K=K,
                      D=D,
                      z_what_dim=z_what_dim,
                      hmc_num_steps=hmc_num_steps,
                      step_size_what=leapfrog_step_size,
                      step_size_where=leapfrog_step_size,
                      leapfrog_num_steps=leapfrog_num_steps,
                      CUDA=CUDA,
                      DEVICE=DEVICE)

    for group, data_path in enumerate(data_paths):
        data = torch.from_numpy(np.load(data_path)).float()
        num_seqs = data.shape[0]
        num_batches = int(num_seqs / batch_size)
        for b in range(num_batches):
            time_start = time.time()
            frames = data[b*batch_size : (b+1)*batch_size].unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
            if CUDA:
                with torch.cuda.device(DEVICE):
                    frames = frames.cuda()
                    mnist_mean = mnist_mean.cuda()

            density_dict = hybrid_objective(model=model,
                                            flags=flags,
                                            AT=AT,
                                            hmc=hmc_sampler,
                                            resampler=resampler,
                                            apg_sweeps=apg_sweeps,
                                            frames=frames,
                                            mnist_mean=mnist_mean,
                                            K=K)
            if flags['apg']:
                metrics['apg'].append(density_dict['apg'][-1].mean().cpu())
            if flags['hmc']:
                metrics['hmc'].append(density_dict['hmc'][-1].mean().cpu())
            if flags['bpg']:
                metrics['bpg'].append(density_dict['bpg'][-1].mean().cpu())
            time_end = time.time()
            print('%d / %d in group %d completed (%ds)' % (b+1, num_batches, group, time_end - time_start))
    # if flags['apg']:
    #     np.save('log_joint_apg_%d' % apg_sweeps, torch.Tensor(metrics['apg']).data.numpy())
    # if flags['hmc']:
    #     np.save('log_joint_hmc_%d' % apg_sweeps, torch.Tensor(metrics['hmc']).data.numpy())
    # if flags['bpg']:
    #     np.save('log_joint_bpg_%d' % apg_sweeps, torch.Tensor(metrics['bpg']).data.numpy())
    return metrics
