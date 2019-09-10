import torch
import time
from utils import *
from ag_ep_pcg_dec import *
from os_ep_dec import *


def test(models, optimizer, OB, mcmc_size, S, B, K, CUDA, device, PATH):
    NUM_DATASETS = OB.shape[0]
    NUM_BATCHES = int((NUM_DATASETS / B))
    (NUM_EPOCHS, NUM_DATASETS, S, B, CUDA, device, path) = Train_Params
    SubTrain_Params = (device, S, B) + Model_Params
    ##(N, K, DCharles Levine, mcmc_size) = Model_Params
    indices = torch.randperm(NUM_DATASETS)
    batch_indices = indices[0*B : (0+1)*B]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(S, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    loss, metric_step, reused = objective(models, obs, SubTrain_Params)
    return obs, metric_step, reused