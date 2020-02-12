import sys
sys.path.append('../')
import torch
from utils import shuffler
from hybrid_objective import hybrid_objective
from resample import Resampler
import time
import numpy as np
from hmc_objective import HMC
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

def test_hybrid_all(model, flags, DATA, batch_size, sample_size, apg_sweeps, K, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, filename, CUDA, DEVICE):
    """
    ==========
    run apg sampler for each of the selected datasets
    ==========

    """
    metrics = {'apg' : [], 'bps': [], 'hmc' : []}
    # metrics_apg = []
    # metrics_hmc = []
    num_datasets = DATA.shape[0]
    N = DATA.shape[1]
    num_batches = int(num_datasets / batch_size)
    (_, enc_apg_local, _, dec) = model
    resampler = Resampler(strategy='systematic',
                          sample_size=sample_size,
                          CUDA=CUDA,
                          DEVICE=DEVICE)

    resampler_bps = Resampler(strategy='systematic',
                              sample_size=sample_size,
                              CUDA=CUDA,
                              DEVICE=DEVICE)

    hmc = HMC(enc_local=enc_apg_local,
              dec=dec,
              burn_in=None,
              S=sample_size,
              B=batch_size,
              N=N,
              K=K,
              D=2,
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
                                         hmc=hmc,
                                         resampler=resampler,
                                         resampler_bps=resampler_bps,
                                         apg_sweeps=apg_sweeps,
                                         ob=ob,
                                         K=K,
                                         hmc_num_steps=hmc_num_steps,
                                         leapfrog_step_size=leapfrog_step_size,
                                         leapfrog_num_steps=leapfrog_num_steps)
        if flags['apg']:
            metrics['apg'].append(density_dict['apg'][-1].mean().cpu())
        if flags['hmc']:
            metrics['hmc'].append(density_dict['hmc'][-1].mean().cpu())
        if flags['bps']:
            metrics['bps'].append(density_dict['bps'][-1].mean().cpu())
        time_end = time.time()
        print('%d / %d completed (%ds)' % (b+1, num_batches, time_end - time_start))
    return metrics



def test_hybrid(model, flags, datas, sample_size, apg_sweeps, K, hmc_num_steps, leapfrog_step_size, leapfrog_num_steps, filename, CUDA, DEVICE):
    """
    ==========
    run apg sampler for each of the selected datasets
    ==========

    """


    for data in datas:
        data = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = data.cuda().to(DEVICE)
        S, B, N, D = ob.shape


        resampler = Resampler(strategy='systematic',
                              sample_size=sample_size,
                              CUDA=CUDA,
                              DEVICE=DEVICE)

        resampler_bps = Resampler(strategy='systematic',
                                  sample_size=sample_size*100,
                                  CUDA=CUDA,
                                  DEVICE=DEVICE)

        (_, enc_apg_local, _, dec) = model
        hmc = HMC(enc_local=enc_apg_local,
                  dec=dec,
                  burn_in=None,
                  S=S,
                  B=B,
                  N=N,
                  K=K,
                  D=D,
                  CUDA=CUDA,
                  DEVICE=DEVICE)

        density_dict = hybrid_objective(model=model,
                                         flags=flags,
                                         hmc=hmc,
                                         resampler=resampler,
                                         resampler_bps=resampler_bps,
                                         apg_sweeps=apg_sweeps,
                                         ob=ob,
                                         K=K,
                                         hmc_num_steps=hmc_num_steps,
                                         leapfrog_step_size=leapfrog_step_size,
                                         leapfrog_num_steps=leapfrog_num_steps)
        if flags['apg']:
            density_apg = density_dict['apg'].cpu().squeeze(-1)
            np.save('log_joint_apg_%s' % filename, density_apg.data.numpy())
        if flags['hmc']:
            density_hmc = density_dict['hmc'].cpu().squeeze(-1)
            np.save('log_joint_hmc_%s' % filename, density_hmc.data.numpy())
        if flags['bps']:
            density_bps = density_dict['bps'].cpu().squeeze(-1)
            np.save('log_joint_bps_%s' % filename, density_bps.data.numpy())

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
                ess.append(trace['ess_mu'][-1] / float(sample_size))
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
