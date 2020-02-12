import torch
import sys
import torch.nn.functional as F
sys.path.append('../')
from utils import shuffler
from hybrid_objective import hybrid_objective
from hmc_objective import HMC
from resample import Resampler
import time
import numpy as np
"""
==========
evaluation functions for apg samplers
==========
"""
def sample_data_uniform(DATAs, data_ptr):
    """
    ==========
    randomly select one gmm dataset from each group,
    given the data pointer
    ==========
    """
    datas = []
    for DATA in DATAs:
        datas.append(DATA[data_ptr])
    return datas

def test_hybrid_all(model, flags, DATA, batch_size, sample_size, apg_sweeps, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, filename, CUDA, DEVICE):
    """
    ==========
    run multiple samplers for each of the selected datasets
    ==========

    """
    metrics = {'apg' : [], 'gibbs' : [], 'hmc' : [], 'bps': []}
    # metrics_apg = []
    # metrics_hmc = []
    num_datasets = DATA.shape[0]
    N = DATA.shape[1]
    num_batches = int(num_datasets / batch_size)
    for b in range(num_batches):
        time_start = time.time()
        ob = DATA[b*batch_size : (b+1)*batch_size]
        ob = ob.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = ob.cuda().to(DEVICE)
        (_, _, _, generative) = model
        S, B, N, D = ob.shape
        hmc_sampler = HMC(generative=generative,
                            burn_in=None,
                            S=sample_size,
                            B=B,
                            N=N,
                            K=3,
                            D=D,
                            CUDA=CUDA,
                            DEVICE=DEVICE)

        resampler = Resampler(strategy='systematic',
                              sample_size=sample_size,
                              CUDA=CUDA,
                              DEVICE=DEVICE)

        resampler_bps = Resampler(strategy='systematic',
                                  sample_size=sample_size,
                                  CUDA=CUDA,
                                  DEVICE=DEVICE)

        density_dict = hybrid_objective(model=model,
                                        flags=flags,
                                        hmc=hmc_sampler,
                                        resampler=resampler,
                                        resampler_bps=resampler_bps,
                                        apg_sweeps=apg_sweeps,
                                        ob=ob,
                                        hmc_num_steps=hmc_num_steps,
                                        leapfrog_step_size=leapfrog_step_size,
                                        leapfrog_num_steps=leapfrog_num_steps)
        if flags['apg']:
            metrics['apg'].append(density_dict['apg'][-1].mean().cpu())
        if flags['hmc']:
            metrics['hmc'].append(density_dict['hmc'][-1].mean().cpu())
        if flags['gibbs']:
            metrics['gibbs'].append(density_dict['gibbs'][-1].mean().cpu())
        if flags['bps']:
            metrics['bps'].append(density_dict['bps'][-1].mean().cpu())
        time_end = time.time()
        print('%d / %d completed (%ds)' % (b+1, num_batches, time_end - time_start))
    return metrics

def test_hybrid(model, flags, datas, sample_size, apg_sweeps, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, filename, CUDA, DEVICE):
    """
    ==========
    run multiple samplers for each of the selected datasets
    ==========

    """

    # metrics_apg = []
    # metrics_hmc = []
    for data in datas:
        data = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = data.cuda().to(DEVICE)
        (_, _, _, generative) = model
        S, B, N, D = ob.shape
        hmc_sampler = HMC(generative=generative,
                            burn_in=None,
                            S=sample_size,
                            B=B,
                            N=N,
                            K=3,
                            D=D,
                            CUDA=CUDA,
                            DEVICE=DEVICE)

        resampler = Resampler(strategy='systematic',
                              sample_size=sample_size,
                              CUDA=CUDA,
                              DEVICE=DEVICE)

        resampler_bps = Resampler(strategy='systematic',
                                  sample_size=sample_size*100,
                                  CUDA=CUDA,
                                  DEVICE=DEVICE)

        density_dict = hybrid_objective(model=model,
                                        flags=flags,
                                        hmc=hmc_sampler,
                                        resampler=resampler,
                                        resampler_bps=resampler_bps,
                                        apg_sweeps=apg_sweeps,
                                        ob=ob,
                                        hmc_num_steps=hmc_num_steps,
                                        leapfrog_step_size=leapfrog_step_size,
                                        leapfrog_num_steps=leapfrog_num_steps)
        if flags['apg']:
            density_apg = density_dict['apg'].cpu().squeeze(-1)
            np.save('log_joint_apg_%s' % filename, density_apg.data.numpy())
        if flags['hmc']:
            density_hmc = density_dict['hmc'].cpu().squeeze(-1)
            np.save('log_joint_hmc_%s' % filename, density_hmc.data.numpy())
        if flags['gibbs']:
            density_gibbs = density_dict['gibbs'].cpu().squeeze(-1)
            np.save('log_joint_gibbs_%s' % filename, density_gibbs.data.numpy())
        if flags['bps']:
            density_bps = density_dict['bps'].cpu().squeeze(-1)
            np.save('log_joint_bps_%s' % filename, density_bps.data.numpy())

#
# def test_budget(model, budget, apg_sweeps, datas, K, CUDA, DEVICE):
#     """
#     ==========
#     compute the ess and log joint under same budget
#     ==========
#     """
#     ESSs = []
#     DENSITIES = []
#
#     loss_required = False
#     ess_required = True
#     mode_required = False
#     density_required = True
#
#     for apg_sweep in apg_sweeps:
#         time_start = time.time()
#         sample_size = int(budget / apg_sweep)
#         resampler = Resampler(strategy='systematic',
#                               sample_size=sample_size,
#                               CUDA=CUDA,
#                               DEVICE=DEVICE)
#
#         ess = []
#         density = []
#         for data in datas:
#             data = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
#             if CUDA:
#                 ob = data.cuda().to(DEVICE)
#             trace = apg_objective(model=model,
#                                   resampler=resampler,
#                                   apg_sweeps=apg_sweep-1,
#                                   ob=ob,
#                                   loss_required=loss_required,
#                                   ess_required=ess_required,
#                                   mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
#                                   density_required=density_required)
#             density.append(trace['density'][-1])
#             if apg_sweep == 1:
#                 ess.append(trace['ess_rws'][0] / float(sample_size))
#
#             else:
#                 ess.append(trace['ess_eta'][-1] / float(sample_size))
#         ESSs.append(ess)
#
#         DENSITIES.append(density)
#         time_end = time.time()
#         print('apg_sweep=%d completed in %ds' % (apg_sweep, time_end - time_start))
#
#     return ESSs, DENSITIES
