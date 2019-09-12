import torch
import time
from utils import *
from normal_gamma import *
from forward_backward_test import *

def Test(models, objective, data, Model_Params, Test_Params):
    """
    compute stepwise elbo as metric
    """

    (NUM_DATASETS, S, B, CUDA, device, path) = Test_Params

    NUM_BATCHES = int((NUM_DATASETS / B))

    SubTest_Params = (device, S, B) + Model_Params
    rand_index = torch.randperm(NUM_DATASETS)
    time_start = time.time()
    dataset_index = rand_index[0]
    obs = data[dataset_index].unsqueeze(0)
    obs = shuffler(obs).repeat(S, 1, 1, 1)
    if CUDA:
            obs =obs.cuda().to(device)
    DBs_eta, DBs_z, ELBOs, Ratios, LLs = objective(models, obs, SubTest_Params)
    return DBs_eta, DBs_z, ELBOs, Ratios, LLs
