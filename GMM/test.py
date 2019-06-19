import torch
import time
from utils import *
from normal_gamma import *
from forward_backward import *

def test_propagation(models, objective, data, Model_Params, Train_Params):
    """
    generic training function
    """
    KLs_propagation = {"kl_eta_ex" : [],"kl_eta_in" : [],"kl_z_ex" : [],"kl_z_in" : []}
    Metrics = {"symKL_DB_eta" : [], "symKL_DB_z" : [], "ess" : []}
    (NUM_DATASETS, S, B, CUDA, device, path) = Train_Params

    NUM_BATCHES = int((NUM_DATASETS / B))
    EPS = torch.FloatTensor([1e-15]).log() ## EPS for KL between categorial distributions
    if CUDA:
        EPS = EPS.cuda().to(device) ## EPS for KL between categorial distributions
    SubTrain_Params = (EPS, device, S, B) + Model_Params
    indices = torch.randperm(NUM_DATASETS)
    time_start = time.time()
    for step in range(NUM_BATCHES):
        batch_indices = indices[step*B : (step+1)*B]
        obs = data[batch_indices]
        obs = shuffler(obs).repeat(S, 1, 1, 1)
        if CUDA:
            obs =obs.cuda().to(device)
        metric_step, reused = objective(models, obs, SubTrain_Params)
        ## gradient step

        for key in Metrics.keys():
            if Metrics[key] == None:
                Metrics[key] = [metric_step[key].cpu().data.numpy()]
            else:
                Metrics[key].append(metric_step[key].cpu().data.numpy())

        if step % 100 == 0:
            time_end = time.time()
            print('iteration:%d/%d' % (step, NUM_BATCHES))
            time_start = time.time()
    return Metrics, reused

def test(models, objective, data, Model_Params, Train_Params):
    """
    generic training function
    """
    KLs_propagation = {"kl_eta_ex" : [],"kl_eta_in" : [],"kl_z_ex" : [],"kl_z_in" : []}
    Metrics = {"symKL_DB_eta" : [], "symKL_DB_z" : [], "ess" : []}
    (NUM_DATASETS, S, B, CUDA, device, path) = Train_Params

    NUM_BATCHES = int((NUM_DATASETS / B))
    EPS = torch.FloatTensor([1e-15]).log() ## EPS for KL between categorial distributions
    if CUDA:
        EPS = EPS.cuda().to(device) ## EPS for KL between categorial distributions
    SubTrain_Params = (EPS, device, S, B) + Model_Params
    indices = torch.randperm(NUM_DATASETS)
    time_start = time.time()
    batch_indices = indices[0*B : (0+1)*B]
    obs = data[batch_indices]
    obs = shuffler(obs).repeat(S, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    _, reused = objective(models, obs, SubTrain_Params)
    return obs, reused
