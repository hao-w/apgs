import torch
import time
from utils import *

def train_baseline(objective, optimizer, data, Train_Params, Model_Params, models):
    """
    generic training function
    """
    (NUM_EPOCHS, NUM_BATCHES, NUM_SEQS, CUDA, device, path) = Train_Params
    (N, K, D, sample_size, batch_size) = Model_Params
    for epoch in range(NUM_EPOCHS):
        metrics = dict()
        time_start = time.time()
        indices = torch.randperm(NUM_SEQS)
        for step in range(NUM_BATCHES):
            optimizer.zero_grad()
            batch_indices = indices[step*batch_size : (step+1)*batch_size]
            obs = data[batch_indices]
            obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
            if CUDA:
                obs =obs.cuda().to(device)
            loss, metric_step, reused = objective(obs, Model_Params, device, models)
            ## gradient step
            loss.backward()
            optimizer.step()
            for key in metric_step.keys():
                if key in metrics:
                    metrics[key] += metric_step[key]
                else:
                    metrics[key] = metric_step[key]
        time_end = time.time()
        metrics_print = ",  ".join(['%s: %.3f' % (k, float(v)/NUM_BATCHES) for k, v in metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        print(metrics_print, file=flog)
        flog.close()
        print("epoch: %d\\%d (%ds),  " % (epoch, NUM_EPOCHS, time_end - time_start) + metrics_print)

def test_baseline(objective, Data, Train_Params, Model_Params, models):
    (NUM_EPOCHS, NUM_BATCHES, NUM_SEQS, CUDA, device, path) = Train_Params
    (N, K, D, sample_size, batch_size) = Model_Params
    indices = torch.randperm(NUM_SEQS)
    batch_indices = indices[0*batch_size : (0+1)*batch_size]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    loss, metric_step, reused = objective(obs, Model_Params, device, models)
    return obs, metric_step, reused
