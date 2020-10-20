import torch
import sys
sys.path.append('../')
from utils import shuffler
from gibbs_objective import gibbs_objective
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

def test_single(model, gibbs_sweeps, datas, sample_size, CUDA, DEVICE):
    """
    ==========
    run apg sampler for each of the selected datasets
    ==========

    """
    mode_required = True
    density_required = True

    metrics = []
    for data in datas:
        data = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = data.cuda().to(DEVICE)
        trace = gibbs_objective(model=model,
                                gibbs_sweeps=gibbs_sweeps,
                                ob=ob,
                                mode_required=mode_required,
                                density_required=density_required)
        metrics.append(trace)
    return metrics
