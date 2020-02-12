import sys
sys.path.append('../')
import torch
import numpy as np
from random import shuffle
from resample import Resampler
from apg_objective import apg_objective
import time
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

def test_all(model, AT, apg_sweeps, data_paths, mnist_mean_path, T, K, D, z_what_dim, batch_size, sample_size, CUDA, DEVICE):
    metrics = dict()
    metrics['mse'] = []
    loss_required = False
    ess_required = False
    mode_required = True
    density_required = True
    mnist_mean = torch.from_numpy(np.load(mnist_mean_path)).float().unsqueeze(0).repeat(K, 1, 1).unsqueeze(0).unsqueeze(0).repeat(sample_size, batch_size, 1, 1, 1)

    resampler = Resampler(strategy='systematic',
                          sample_size=sample_size,
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

            trace = apg_objective(model=model,
                                    resampler=resampler,
                                    AT=AT,
                                    apg_sweeps=apg_sweeps,
                                    frames=frames,
                                    mnist_mean=mnist_mean,
                                    K=K,
                                    loss_required=loss_required,
                                    ess_required=ess_required,
                                    mode_required=mode_required,
                                    density_required=density_required)
            mse = ((trace['E_recon'].cpu() - frames.cpu()) ** 2).mean(1).mean(1).sum(-1).sum(-1).sum(-1)
            metrics['mse'].append(mse / T)
            time_end = time.time()
            print('%d / %d in group %d completed (%ds)' % (b+1, num_batches, group, time_end - time_start))
            if b == 10:
                break
#     np.save('mse_apg_T=%d_D=%d' % (T, K), torch.Tensor(metrics['mse']).data.numpy())

    return metrics



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

    resampler = Resampler(strategy='systematic',
                          sample_size=sample_size,
                          CUDA=CUDA,
                          DEVICE=DEVICE)

    frames = frames.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
    if CUDA:
        with torch.cuda.device(DEVICE):
            frames = frames.cuda()
            mnist_mean = mnist_mean.cuda()

    trace = apg_objective(model=model,
                          resampler=resampler,
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
