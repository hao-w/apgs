import torch
import time
import numpy as np

def train(models, objective, optimizer, data_path, mcmc_steps, mnist_mean, crop, Train_Params, CUDA, device, path):
    (NUM_EPOCHS, NUM_GROUPS, K, D, S, B) = Train_Params
    """
    training function
    """

    # tj_std = torch.ones(D) * 0.1
    if CUDA:
        with torch.cuda.device(device):
            mnist_mean = mnist_mean.cuda().unsqueeze(0).unsqueeze(0).repeat(S, B, 1, 1)
    for epoch in range(NUM_EPOCHS):
        Metrics = dict()
        time_start = time.time()
        group_ind = torch.randperm(NUM_GROUPS)
        for g in range(NUM_GROUPS):
            data = torch.from_numpy(np.load(data_path + 'ob_%d.npy' % group_ind[g])).float()
            # tjs = torch.from_numpy(np.load(data_path + 'tj_%d.npy' % group_ind[g])).float()
            NUM_DATASETS = data.shape[0]
            NUM_BATCHES = int((NUM_DATASETS / B))
            indices = torch.randperm(NUM_DATASETS)
            for step in range(NUM_BATCHES):
                optimizer.zero_grad()
                b_ind = indices[step*B : (step+1)*B]
                frames = data[b_ind] ## B * T * 64 * 64
                # tj = tjs[b_ind].transpose(1,2)
                if CUDA:
                    with torch.cuda.device(device):
                        frames = frames.cuda()
                        # tj = tj.cuda()
                        # tj_std = tj_std.cuda()
                # print(tj.shape)

                metrics = objective(models, K, D, frames, mcmc_steps, mnist_mean, crop)
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
        metrics_print = ",  ".join(['%s: %.3f' % (k, v/(NUM_GROUPS*NUM_BATCHES)) for k, v in Metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        time_end = time.time()
        print("(%ds) " % (time_end - time_start) + metrics_print, file=flog)
        flog.close()
        print("Completed  in (%ds),  " % (time_end - time_start) + metrics_print)
