import sys
sys.path.append('../')
import torch
import numpy as np
from random import shuffle
from resample import resample
from apg_objective import apg_objective

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

def test_single(model, AT, apg_sweeps, frames, mnist_mean_path, K, sample_size, CUDA, DEVICE):
    """
    ==========
    run apg sampler for each of the selected datasets
    ==========

    """
    loss_required = True
    ess_required = True
    mode_required = True
    density_required = True
    mnist_mean = torch.from_numpy(np.load(mnist_mean_path)).float().repeat(K, 1, 1).unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
    metrics = []
    frames = frames.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
    if CUDA:
        with torch.cuda.device(DEVICE):
            frames = frames.cuda()
            mnist_mean = mnist_mean.cuda()
    trace = apg_objective(model=model,
                          resample=resample,
                          AT=AT,
                          apg_sweeps=apg_sweeps,
                          frames=frames,
                          mnist_mean=mnist_mean,
                          K=K,
                          loss_required=loss_required,
                          ess_required=ess_required,
                          mode_required=mode_required,
                          density_required=density_required)
    return trace
