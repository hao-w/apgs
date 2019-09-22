import torch
import time
from utils import *
from normal_gamma import *
from kls import *

def train(models, objective, optimizer, data, mcmc_steps, Train_Params):
    """
    generic training function
    data is a list of datasets
    """
    (NUM_EPOCHS, K, D, S, B, CUDA, device, path) = Train_Params
    GROUP_SIZE = len(data)
    NUM_DATASETS = data[0].shape[0]
    NUM_BATCHES = int((NUM_DATASETS / B))
    annealed_coefficient = (torch.arange(mcmc_steps+1) + 1).float() / (mcmc_steps+1)
    EPS = torch.FloatTensor([1e-15]).log() ## EPS for KL between categorial distributions
    if CUDA:
        EPS = EPS.cuda().to(device) ## EPS for KL between categorial distributions
    for epoch in range(NUM_EPOCHS):
        Metrics = dict()
        time_start = time.time()
        GROUP_ind = torch.randperm(GROUP_SIZE)
        print("epoch=%d\\%d, data order=%s" % (epoch, NUM_EPOCHS, GROUP_ind))
        for g in range(GROUP_SIZE):
            data_g = data[GROUP_ind[g]]
            indices = torch.randperm(NUM_DATASETS)
            for step in range(NUM_BATCHES):
                optimizer.zero_grad()
                batch_indices = indices[step*B : (step+1)*B]
                ob = data_g[batch_indices]
                ob = shuffler(ob).repeat(S, 1, 1, 1)
                if CUDA:
                    with torch.cuda.device(device):
                        ob = ob.cuda()
                        annealed_coefficient = annealed_coefficient.cuda()

                metrics, reused = objective(models, ob, mcmc_steps)
                loss = (torch.cat(metrics['loss'], 0) * annealed_coefficient).sum()
                # loss = torch.cat(metrics['loss'], 0).sum()
                ## gradient step
                loss.backward()
                optimizer.step()
                for key in metrics.keys():
                    if key in Metrics:
                        Metrics[key] += metrics[key][-1].detach().item()
                    else:
                        Metrics[key] = metrics[key][-1].detach().item()
                ## compute KL
                kl_step = kl_train(models, ob, reused, EPS)
                for key in kl_step.keys():
                    if key in Metrics:
                        Metrics[key] += kl_step[key]
                    else:
                        Metrics[key] = kl_step[key]
        time_end = time.time()
        metrics_print = ",  ".join(['%s: %.3f' % (k, v/(GROUP_SIZE*NUM_BATCHES)) for k, v in Metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        print("(%ds) " % (time_end - time_start) + metrics_print, file=flog)
        flog.close()
        print("Completed  in (%ds),  " % (time_end - time_start) + metrics_print)
