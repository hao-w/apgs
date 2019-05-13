import torch
import time
from utils import *

def train_v2(models, objective, optimizer_enc, optimizer_dec, data, Model_Params, Train_Params):
    """
    generic training function
    """
    (NUM_EPOCHS, NUM_DATASETS, S, B, CUDA, device, path) = Train_Params
    SubTrain_Params = (device, S, B) + Model_Params

    NUM_BATCHES = int((NUM_DATASETS / B))
    for epoch in range(NUM_EPOCHS):
        metrics = dict()
        time_start = time.time()
        indices = torch.randperm(NUM_DATASETS)
        for step in range(NUM_BATCHES):
            optimizer_enc.zero_grad()
            batch_indices = indices[step*B : (step+1)*B]
            obs = data[batch_indices]
            obs = shuffler(obs).repeat(S, 1, 1, 1)
            if CUDA:
                obs =obs.cuda().to(device)
            loss, metric_step, reused = objective(models, obs, SubTrain_Params, p_flag=False)
            ## gradient step
            loss.backward()
            optimizer_enc.step()
            ## update generative network
            optimizer_enc.zero_grad()
            loss, metric_step, reused = objective(models, obs, SubTrain_Params, p_flag=True)
            ## gradient step
            loss.backward()
            optimizer_dec.step()

            for key in metric_step.keys():
                if key in metrics:
                    metrics[key] += metric_step[key].mean().item()
                else:
                    metrics[key] = metric_step[key].mean().item()

        time_end = time.time()
        metrics_print = ",  ".join(['%s: %.3f' % (k, v/NUM_BATCHES) for k, v in metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        print(metrics_print, file=flog)
        flog.close()
        print("epoch: %d\\%d (%ds),  " % (epoch, NUM_EPOCHS, time_end - time_start) + metrics_print)


def test_v2(models, objective, Data, Model_Params, Train_Params):
    (NUM_EPOCHS, NUM_DATASETS, S, B, CUDA, device, path) = Train_Params
    SubTrain_Params = (device, S, B) + Model_Params
    ##(N, K, D, mcmc_size) = Model_Params
    indices = torch.randperm(NUM_DATASETS)
    batch_indices = indices[0*B : (0+1)*B]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(S, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    loss, metric_step, reused = objective(models, obs, SubTrain_Params)
    return obs, metric_step, reused
