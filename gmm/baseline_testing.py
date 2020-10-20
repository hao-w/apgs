import torch
import sys
sys.path.append('../')
from utils import shuffler
from baseline_objective import rws_objective
import time
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

def test_single(model, architecture, datas, sample_size, CUDA, DEVICE):
    """
    ==========
    run apg sampler for each of the selected datasets
    ==========

    """
    loss_required = True
    ess_required = True
    mode_required = True
    density_required = True

    metrics = []
    for data in datas:
        data = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = data.cuda().to(DEVICE)
        trace = rws_objective(model=model,
                              ob=ob,
                              architecture=architecture,
                              loss_required=loss_required,
                              ess_required=ess_required,
                              mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                              density_required=density_required)
        metrics.append(trace)
    return metrics

def test_budget(model, budget, apg_sweeps, datas, K, CUDA, DEVICE):
    """
    ==========
    compute the ess and log joint under same budget
    ==========
    """
    ESSs = []
    DENSITIES = []

    loss_required = False
    ess_required = True
    mode_required = False
    density_required = True

    for apg_sweep in apg_sweeps:
        time_start = time.time()
        sample_size = int(budget / apg_sweep)
        resampler = Resampler(strategy='systematic',
                              sample_size=sample_size,
                              CUDA=CUDA,
                              DEVICE=DEVICE)

        ess = []
        density = []
        for data in datas:
            data = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
            if CUDA:
                ob = data.cuda().to(DEVICE)
            trace = apg_objective(model=model,
                                  resampler=resampler,
                                  apg_sweeps=apg_sweep-1,
                                  ob=ob,
                                  loss_required=loss_required,
                                  ess_required=ess_required,
                                  mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                                  density_required=density_required)
            density.append(trace['density'][-1])
            if apg_sweep == 1:
                ess.append(trace['ess_rws'][0] / float(sample_size))

            else:
                ess.append(trace['ess_eta'][-1] / float(sample_size))
        ESSs.append(ess)

        DENSITIES.append(density)
        time_end = time.time()
        print('apg_sweep=%d completed in %ds' % (apg_sweep, time_end - time_start))

    return ESSs, DENSITIES
