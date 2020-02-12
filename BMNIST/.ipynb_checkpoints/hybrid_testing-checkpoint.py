import sys
sys.path.append('../')
import torch
import numpy as np
from random import shuffle
from resample import Resampler
from hybrid_objective import hybrid_objective
import time
import numpy as np
from hmc_objective import HMC
"""
==========
evaluation functions for apg samplers
==========
"""
def sample_data_uniform(DATA_PATHS, data_ptr):
    """
    ==========
    randomly select one bmnist file from the list of data paths
    ==========
    """
    shuffle(DATA_PATHS)
    data = torch.from_numpy(np.load(DATA_PATHS[0])).float() ## load testing datasets
    return data[data_ptr]

def test_hybrid(model, flags, AT, apg_sweeps, frames, mnist_mean_path, K, D, z_what_dim, sample_size, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, filename, CUDA, DEVICE):
    mnist_mean = torch.from_numpy(np.load(mnist_mean_path)).float().repeat(K, 1, 1).unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
    resampler = Resampler(strategy='systematic',
                          sample_size=sample_size,
                          CUDA=CUDA,
                          DEVICE=DEVICE)


    frames = frames.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
    if CUDA:
        with torch.cuda.device(DEVICE):
            frames = frames.cuda()
            mnist_mean = mnist_mean.cuda()
    S, B, T, _, _ = frames.shape
    hmc = HMC(model=model,
              AT=AT,
              burn_in=None,
              S=S,
              B=B,
              T=T,
              K=K,
              D=D,
              z_what_dim=z_what_dim,
              CUDA=CUDA,
              DEVICE=DEVICE)

    density_dict = hybrid_objective(model=model,
                                    flags=flags,
                                    hmc=hmc,
                                    AT=AT,
                                    resampler=resampler,
                                    apg_sweeps=apg_sweeps,
                                    frames=frames,
                                    mnist_mean=mnist_mean,
                                    K=K,
                                    hmc_num_steps=hmc_num_steps,
                                    leapfrog_step_size=leapfrog_step_size,
                                    leapfrog_num_steps=leapfrog_num_steps)
    if flags['apg']:
        density_apg = density_dict['apg'].cpu().squeeze(-1)
        np.save('log_joint_apg_%sweeps_%s' % (apg_sweeps,filename), density_apg.data.numpy())
    if flags['hmc']:
        density_hmc = density_dict['hmc'].cpu().squeeze(-1)
        np.save('log_joint_hmc_%ssweeps_%s' % (apg_sweeps,filename), density_hmc.data.numpy())
    if flags['bps']:
        density_bps = density_dict['bps'].cpu().squeeze(-1)
        np.save('log_joint_bps_%ssweeps_%s' % (apg_sweeps,filename), density_bps.data.numpy())



def test_hybrid_all(model, flags, AT, apg_sweeps, data_paths, mnist_mean_path, T, K, D, z_what_dim, batch_size, sample_size, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, filename, CUDA, DEVICE):
    metrics = {'apg' : [], 'hmc' : [], "bps" : []}
    mnist_mean = torch.from_numpy(np.load(mnist_mean_path)).float().unsqueeze(0).repeat(K, 1, 1).unsqueeze(0).unsqueeze(0).repeat(sample_size, batch_size, 1, 1, 1)

    resampler = Resampler(strategy='systematic',
                          sample_size=sample_size,
                          CUDA=CUDA,
                          DEVICE=DEVICE)
    hmc = HMC(model=model,
              AT=AT,
              burn_in=None,
              S=sample_size,
              B=batch_size,
              T=T,
              K=K,
              D=D,
              z_what_dim=z_what_dim,
              CUDA=CUDA,
              DEVICE=DEVICE)

    for group, data_path in enumerate(data_paths):
#         time_start = time.time()
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
                                            hmc=hmc,
                                            AT=AT,
                                            resampler=resampler,
                                            apg_sweeps=apg_sweeps,
                                            frames=frames,
                                            mnist_mean=mnist_mean,
                                            K=K,
                                            hmc_num_steps=hmc_num_steps,
                                            leapfrog_step_size=leapfrog_step_size,
                                            leapfrog_num_steps=leapfrog_num_steps)
            if flags['apg']:
                metrics['apg'].append(density_dict['apg'][-1].mean().cpu())
            if flags['hmc']:
                metrics['hmc'].append(density_dict['hmc'][-1].mean().cpu())
            if flags['bps']:
                metrics['bps'].append(density_dict['bps'][-1].mean().cpu())
            time_end = time.time()
#             if b % 1 == 0:
            break
            print('%d / %d in group %d completed (%ds)' % (b+1, num_batches, group, time_end - time_start))
    if flags['apg']:
        np.save('log_joint_apg_%d' % apg_sweeps, torch.Tensor(metrics['apg']).data.numpy())
    if flags['hmc']:
        np.save('log_joint_hmc_%d' % apg_sweeps, torch.Tensor(metrics['hmc']).data.numpy())
    if flags['bps']:
        np.save('log_joint_bps_%d' % apg_sweeps, torch.Tensor(metrics['bps']).data.numpy())
    return metrics
