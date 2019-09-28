import torch
import time
from utils import *
import gc

def train(models, objective, optimizer, data, mcmc_steps, Train_Params, CUDA, device, path):
    """
    training function for datasets with various sizes
    """
    (NUM_EPOCHS, K, D, S, B) = Train_Params
    GROUP_SIZE = len(data)
    NUM_DATASETS = data[0].shape[0]
    NUM_BATCHES = int((NUM_DATASETS / B))
    # annealed_coefficient = (torch.arange(mcmc_steps+1) + 1).float() / (mcmc_steps+1)
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
                        # annealed_coefficient = annealed_coefficient.cuda()
                metrics = objective(models, optimizer, ob, mcmc_steps, K)
                phi_loss = torch.cat(metrics['phi_loss'], 0).sum()
                theta_loss = (torch.cat(metrics['theta_loss'], 0)).sum()
                phi_loss.backward(retain_graph=True)
                theta_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                for key in metrics.keys():
                    if key in Metrics:
                        Metrics[key] += metrics[key][-1].detach().item()
                    else:
                        Metrics[key] = metrics[key][-1].detach().item()

        metrics_print = ",  ".join(['%s: %.3f' % (k, v/(GROUP_SIZE*NUM_BATCHES)) for k, v in Metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        time_end = time.time()
        print("(%ds) " % (time_end - time_start) + metrics_print, file=flog)
        flog.close()
        print("Completed  in (%ds),  " % (time_end - time_start) + metrics_print)
