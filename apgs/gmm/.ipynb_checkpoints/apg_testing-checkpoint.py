import torch
import sys
sys.path.append('../')
from utils import shuffler
from apg_objective_small import apg_objective
from resample import Resampler
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
#
# def test_ess(model, DATA, apg_sweeps, sample_size, batch_size, CUDA, DEVICE):
#     """
#     ==========
#     run apg sampler for each of the selected datasets
#     ==========
#
#     """
#     # num_datasets = 100
#     num_datasets = DATA.shape[0]
#     N = DATA.shape[1]
#     num_batches = int((num_datasets / batch_size))
#
#     loss_required = False
#     ess_required = True
#     mode_required = False
#     density_required = False
#     kl_required = True
#     ESS = {'K=2' : [], 'K=5', 'K=10' : []}
#     KL = {'K=2' : [], 'K=5', 'K=10' : []}
#
#     indices = torch.arange(num_datasets)
#     sample_size = int(budget / apg_sweep)
#     resampler = Resampler(strategy='systematic',
#                           sample_size=sample_size,
#                           CUDA=CUDA,
#                           DEVICE=DEVICE)
#
#     for b in range(num_batches):
#         time_start = time.time()
#         batch_indices = indices[b*batch_size : (b+1)*batch_size]
#         ob = shuffler(data[batch_indices]).repeat(sample_size, 1, 1, 1)
#         if CUDA:
#             ob = ob.cuda().to(DEVICE)
#
#         trace = apg_objective(model=model,
#                               resampler=resampler,
#                               apg_sweeps=apg_sweep-1,
#                               ob=ob,
#                               loss_required=loss_required,
#                               ess_required=ess_required,
#                               mode_required=mode_required,
#                               density_required=density_required,
#                               kl_required=kl_required)
#         if apg_sweep == 1:
#             ess.append(trace['ess_rws'][0].unsqueeze(-1))
#         else:
#             ess.append(trace['ess_eta'][-1].unsqueeze(-1))
#         # density.append(trace['density'][-1].unsqueeze(-1))
#     # density = torch.cat(density, 0)
#         time_end = time.time()
#         print('(%ds) %d / %d' % (time_end - time_start, b+1, num_batches))
#     ess = torch.cat(ess, 0)
#
#     # col = 'sweep=%d, sample=%d' % (apg_sweep, sample_size)
#     # ESS[col] = ess.squeeze(-1).cpu().data.numpy() / sample_size
#     # DENSITY[col] = density.squeeze(-1).cpu().data.numpy()
#
#
#     return ESS, DENSITY

def test_single(model, resampler, apg_sweeps, datas, sample_size, CUDA, DEVICE):
    """
    ==========small
    run apg sampler for each of the selected datasets
    ==========

    """
    loss_required = True
    ess_required = True
    mode_required = True
    density_required = True
    kl_required = True

    metrics = []
    for data in datas:
        data = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = data.cuda().to(DEVICE)
        trace = apg_objective(model=model,
                              resampler=resampler,
                              apg_sweeps=apg_sweeps,
                              ob=ob,
                              loss_required=loss_required,
                              ess_required=ess_required,
                              mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                              density_required=density_required,
                              kl_required=kl_required)
        metrics.append(trace)
    return metrics


def test_budget_grid(model, apg_sweeps, sample_sizes, data, K, CUDA, DEVICE):
    """
    ==========
    compute the ess and log joint under same budget
    ==========
    """
    # ESSs = []
    # DENSITIES = []

    loss_required = False
    ess_required = True
    mode_required = False
    density_required = True

    ess = []
    density = []
    for i, apg_sweep in enumerate(apg_sweeps):
        sample_size = sample_sizes[i]
        time_start = time.time()
        # sample_size = int(budget / apg_sweep)
        resampler = Resampler(strategy='systematic',
                              sample_size=sample_size,
                              CUDA=CUDA,
                              DEVICE=DEVICE)

        ob = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = ob.cuda().to(DEVICE)
        trace = apg_objective(model=model,
                              resampler=resampler,
                              apg_sweeps=apg_sweep-1,
                              ob=ob,
                              loss_required=loss_required,
                              ess_required=ess_required,
                              mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                              density_required=density_required)
        density.append(trace['density'][-1].mean(0).cpu())
        ess.append(trace['ess_rws'][-1].cpu() / float(sample_size))

        # if apg_sweep == 1:
        #     ess.append(trace['ess_rws'][0].cpu() / float(sample_size))
        #
        # else:
        #     ess.append(trace['ess_eta'][-1].cpu() / float(sample_size))

        time_end = time.time()
    # print('apg_sweep=%d completed in %ds' % (apg_sweep, time_end - time_start))

    return ess, density





def test_budget(model, budget, apg_sweeps, datas, K, CUDA, DEVICE):
    """
    ==========
    compute the ess and log joint under same budget
    ==========
    """
    # ESSs = []
    # DENSITIES = []

    loss_required = False
    ess_required = True
    mode_required = False
    density_required = True

    ess = []
    density = []
    for apg_sweep in apg_sweeps:
        time_start = time.time()
        sample_size = int(budget / apg_sweep)
        resampler = Resampler(strategy='systematic',
                              sample_size=sample_size,
                              CUDA=CUDA,
                              DEVICE=DEVICE)

        data = datas[0].unsqueeze(0).repeat(sample_size, 1, 1, 1)
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
        density.append(trace['density'][-1].mean(0))
        if apg_sweep == 1:
            ess.append(trace['ess_rws'][0] / float(sample_size))

        else:
            ess.append(trace['ess_eta'][-1] / float(sample_size))

        time_end = time.time()
        # print('apg_sweep=%d completed in %ds' % (apg_sweep, time_end - time_start))

    return ess, density
