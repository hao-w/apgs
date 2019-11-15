import sys
sys.path.append('../')
import torch
from utils import shuffler
from resample import resample
from apg_objective import apg_objective
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

def test_single(model, apg_sweeps, datas, K, sample_size, CUDA, DEVICE):
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
        trace = apg_objective(model=model,
                              resample=resample,
                              apg_sweeps=apg_sweeps,
                              ob=ob,
                              K=K,
                              loss_required=loss_required,
                              ess_required=ess_required,
                              mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                              density_required=density_required)
        metrics.append(trace)
    return metrics
