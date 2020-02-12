import sys
sys.path.append('../')
import torch
from utils import shuffler
from apg_objective import apg_objective
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
    randomly select one dgmm dataset from each group,
    given the data pointer
    ==========
    """
    datas = []
    for DATA in DATAs:
        datas.append(DATA[data_ptr])
    return datas

def test_single(model, resampler, apg_sweeps, datas, K, sample_size, CUDA, DEVICE):
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
                              resampler=resampler,
                              apg_sweeps=apg_sweeps,
                              ob=ob,
                              K=K,
                              loss_required=loss_required,
                              ess_required=ess_required,
                              mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                              density_required=density_required)
        metrics.append(trace)
    return metrics


def test_budget_grid(model, apg_sweeps, sample_sizes, data, K, CUDA, DEVICE):
    """
    ==========
    compute the ess and log joint under same budget
    ==========
    """
    ess = []
    density = []

    loss_required = False
    ess_required = True
    mode_required = False
    density_required = True

    for i, apg_sweep in enumerate(apg_sweeps):

        sample_size = sample_sizes[i]
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
                              K=K,
                              loss_required=loss_required,
                              ess_required=ess_required,
                              mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                              density_required=density_required)
        density.append(trace['density'][-1].cpu())
        if apg_sweep == 1:
            ess.append(trace['ess_rws'][0].cpu() / float(sample_size))
        else:
            ess.append(trace['ess_mu'].mean(0).cpu() / float(sample_size))
    return ess, density


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
        print('sample_size=%d' % sample_size)
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
                                  K=K,
                                  loss_required=loss_required,
                                  ess_required=ess_required,
                                  mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                                  density_required=density_required)
            density.append(trace['density'][-1])
            if apg_sweep == 1:
                ess.append(trace['ess_rws'][0] / float(sample_size))

            else:
                ess.append(trace['ess_mu'].mean(0) / float(sample_size))
        ESSs.append(ess)

        DENSITIES.append(density)
        time_end = time.time()
        print('apg_sweep=%d completed in %ds' % (apg_sweep, time_end - time_start))

    return ESSs, DENSITIES
    # #
    # def Test_ALL(self, objective, Data, mcmc_steps, Test_Params):
    #     Metrics = {'kl_ex' : [], 'kl_in' : [], 'll' : [], 'ess' : []}
    #     (S, B, DEVICE) = Test_Params
    #     EPS = torch.FloatTensor([1e-15]).log() ## EPS for KL between categorial distributions
    #     EPS = EPS.cuda().to(DEVICE) ## EPS for KL between categorial distributions
    #     for g, data_g in enumerate(Data):
    #         print("group=%d" % (g+1))
    #         NUM_DATASETS = data_g.shape[0]
    #         NUM_BATCHES = int((NUM_DATASETS / B))
    #         for step in range(NUM_BATCHES):
    #             ob = data_g[step*B : (step+1)*B]
    #             ob = shuffler(ob).repeat(S, 1, 1, 1)
    #             ob = ob.cuda().to(DEVICE)
    #             metrics = objective(self.models, ob, mcmc_steps, EPS)
    #             Metrics['ess'].append(metrics['ess'])
    #             Metrics['ll'].append(metrics['ll'])
    #             Metrics['kl_ex'].append(metrics['kl_ex'])
    #             Metrics['kl_in'].append(metrics['kl_in'])
    #
    #     return Metrics
