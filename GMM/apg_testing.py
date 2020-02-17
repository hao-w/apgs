import torch
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils import shuffler
from apg_objective import apg_objective
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
    selected_data = []
    for DATA in DATAs:
        selected_data.append(DATA[data_ptr])
    return selected_data

def test_single(model, block, apg_sweeps, datas, sample_size, CUDA, DEVICE):
    """
    ==========
    run apg sampler for each of the selected datasets
    ==========

    """
    loss_required = True
    ess_required = True
    mode_required = True
    density_required = True
    kl_required = True

    metrics = []

    resampler = Resampler(strategy='systematic',
                          sample_size=sample_size,
                          CUDA=CUDA,
                          DEVICE=DEVICE)
    for data in datas:
        data = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = data.cuda().to(DEVICE)
        trace = apg_objective(model=model,
                              block=block,
                              resampler=resampler,
                              apg_sweeps=apg_sweeps,
                              ob=ob,
                              loss_required=loss_required,
                              ess_required=ess_required,
                              mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                              density_required=density_required)
        metrics.append(trace)
    return metrics

def test_budget_grid(model, apg_sweeps, batch_size, sample_sizes, DATA, K, CUDA, DEVICE):
    """
    ==========
    compute the ess and log joint under same budget
    ==========
    """
    loss_required = False
    ess_required = True
    mode_required = False
    density_required = True

    metrics = dict()
    num_datasets, N, D = DATA.shape
    num_batches = int(num_datasets / batch_size)
    for i, apg_sweep in enumerate(apg_sweeps):
        metrics_one_run = {'ess_small' : [], 'density_small' : [], 'ess_large' : [], 'density_large' : []}
        time_start = time.time()
        sample_size = sample_sizes[i]
        time_start = time.time()
        resampler = Resampler(strategy='systematic',
                              sample_size=sample_size,
                              CUDA=CUDA,
                              DEVICE=DEVICE)
        for b in range(num_batches):
            ob = DATA[b*batch_size : (b+1)*batch_size]
            ob = ob.unsqueeze(0).repeat(sample_size, 1, 1, 1)
            if CUDA:
                ob = ob.cuda().to(DEVICE)

            trace = apg_objective(model=model,
                                  block='small',
                                  resampler=resampler,
                                  apg_sweeps=apg_sweep-1,
                                  ob=ob,
                                  loss_required=loss_required,
                                  ess_required=ess_required,
                                  mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                                  density_required=density_required)
            metrics_one_run['density_small'].append(trace['density'][-1].mean().cpu())
            metrics_one_run['ess_small'].append(trace['ess'][-1].mean().cpu() / float(sample_size))

            trace = apg_objective(model=model,
                                  block='large',
                                  resampler=resampler,
                                  apg_sweeps=apg_sweep-1,
                                  ob=ob,
                                  loss_required=loss_required,
                                  ess_required=ess_required,
                                  mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                                  density_required=density_required)
            metrics_one_run['density_large'].append(trace['density'][-1].mean().cpu())
            metrics_one_run['ess_large'].append(trace['ess'][-1].mean().cpu() / float(sample_size))
        for key in metrics_one_run.keys():
            if key in metrics:
                metrics[key].append(torch.Tensor(metrics_one_run[key]).mean())
            else:
                metrics[key] = [torch.Tensor(metrics_one_run[key]).mean()]

        time_end = time.time()
        print('K=%d, L=%d completed in %ds' % (apg_sweep, sample_size, time_end - time_start))
    return metrics
