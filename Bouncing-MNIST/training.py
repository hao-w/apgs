import torch
import time
import numpy as np
from model_operations import Save_models

def train(APG, optimizer, data_path, Train_Params, APG_STEPS, DEVICE, PATH):
    (NUM_EPOCHS, NUM_GROUPS, S, B) = Train_Params
    """
    training function
    """
    for epoch in range(NUM_EPOCHS):
        group_ind = torch.randperm(NUM_GROUPS)
        for g in range(NUM_GROUPS):
            time_start = time.time()
            Metrics = dict()
            data = torch.from_numpy(np.load(data_path + 'ob_%d.npy' % group_ind[g])).float()
            NUM_DATASETS = data.shape[0]
            NUM_BATCHES = int((NUM_DATASETS / B))
            indices = torch.randperm(NUM_DATASETS)
            for step in range(NUM_BATCHES):
                optimizer.zero_grad()
                b_ind = indices[step*B : (step+1)*B]
                frames = data[b_ind].repeat(S, 1, 1, 1, 1).cuda().to(DEVICE) ## S * B * T * 64 * 64
                metrics = APG.Sweeps(APG_STEPS, S, B, frames)
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
            metrics_print = ",  ".join(['%s: %.3f' % (k, v/(NUM_BATCHES)) for k, v in Metrics.items()])
            flog = open('../results/log-' + PATH + '.txt', 'a+')
            time_end = time.time()
            print("Epoch=%d, Group=%d, (%ds) " % (epoch, g, time_end - time_start) + metrics_print, file=flog)
            flog.close()
            print("Epoch=%d, Group=%d, (%ds),  " % (epoch, g, time_end - time_start) + metrics_print)

            Save_models(APG.models, PATH=PATH)
