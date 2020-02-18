import sys
sys.path.append('../')
import torch
from utils import shuffler
from hybrid_objective import hybrid_objective
from resample import Resampler
import time
import numpy as np
from hmc_objective import HMC
import os

"""
==========
evaluation functions for apg samplers
==========
"""
def sample_data_uniform(DATAs, data_ptr):
    """
    ==========
    randomly select one dgmm dataset from each group,
    given the data pointer
    ==========
    """
    datas = []
    for DATA in DATAs:
        datas.append(DATA[data_ptr])
    return datas

def test_hybrid(num_runs, model, flags, data, sample_size, apg_sweeps, K, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, CUDA, DEVICE):
    DENSITIES = {'apg' : [], 'hmc' : [], 'bpg' : []}
    data = data.unsqueeze(0).unsqueeze(0).repeat(sample_size, 1, 1, 1)
    if CUDA:
        ob = data.cuda().to(DEVICE)
    S, B, N, D = ob.shape
    (_, enc_apg_local, _, dec) = model
    hmc_sampler = HMC(enc_local=enc_apg_local,
                      dec=dec,
                      S=S,
                      B=B,
                      N=N,
                      K=K,
                      D=D,
                      hmc_num_steps=hmc_num_steps,
                      leapfrog_step_size=leapfrog_step_size,
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
                                     hmc=hmc_sampler,
                                     resampler=resampler,
                                     apg_sweeps=apg_sweeps,
                                     ob=ob,
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
    if flags['apg']:
        np.save('log_joint_apg', np.concatenate(DENSITIES['apg'], 0))
    if flags['hmc']:
        np.save('log_joint_hmc', np.concatenate(DENSITIES['hmc'], 0))
    if flags['bpg']:
        np.save('log_joint_bpg', np.concatenate(DENSITIES['bpg'], 0))

def test_hybrid_all(model, flags, DATA, batch_size, sample_size, apg_sweeps, K, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, CUDA, DEVICE):
    metrics = {'apg' : [], 'bpg': [], 'hmc' : []}
    num_datasets, N, D = DATA.shape
    num_batches = int(num_datasets / batch_size)
    (_, enc_apg_local, _, dec) = model
    resampler = Resampler(strategy='systematic',
                          sample_size=sample_size,
                          CUDA=CUDA,
                          DEVICE=DEVICE)

    hmc_sampler = HMC(enc_local=enc_apg_local,
                      dec=dec,
                      S=sample_size,
                      B=batch_size,
                      N=N,
                      K=K,
                      D=2,
                      hmc_num_steps=hmc_num_steps,
                      leapfrog_step_size=leapfrog_step_size,
                      leapfrog_num_steps=leapfrog_num_steps,
                      CUDA=CUDA,
                      DEVICE=DEVICE)

    for b in range(num_batches):
        time_start = time.time()
        ob = DATA[b*batch_size : (b+1)*batch_size]
        ob = ob.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = ob.cuda().to(DEVICE)

        density_dict = hybrid_objective(model=model,
                                         flags=flags,
                                         hmc=hmc_sampler,
                                         resampler=resampler,
                                         apg_sweeps=apg_sweeps,
                                         ob=ob,
                                         K=K)
        if flags['apg']:
            metrics['apg'].append(density_dict['apg'][-1].mean().cpu())
        if flags['hmc']:
            metrics['hmc'].append(density_dict['hmc'][-1].mean().cpu())
        if flags['bpg']:
            metrics['bpg'].append(density_dict['bpg'][-1].mean().cpu())
        time_end = time.time()
        print('%d / %d completed (%ds)' % (b+1, num_batches, time_end - time_start))
    return metrics
