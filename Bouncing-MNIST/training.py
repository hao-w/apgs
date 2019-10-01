import torch
import time

def train(models, objective, optimizer, data, tjs, mcmc_steps, mnist_mean, crop, Train_Params, CUDA, device, path):
    (NUM_EPOCHS, T, K, D, S, B) = Train_Params
    """
    training function
    """
    NUM_DATASETS = data.shape[0]
    NUM_BATCHES = int((NUM_DATASETS / B))
    tj_std = torch.ones(D) * 0.1
    if CUDA:
        with torch.cuda.device(device):
            mnist_mean = mnist_mean.cuda().unsqueeze(0).unsqueeze(0).repeat(S, B, 1, 1)
    for epoch in range(NUM_EPOCHS):
        Metrics = dict()
        time_start = time.time()
        indices = torch.randperm(NUM_DATASETS)
        for step in range(NUM_BATCHES):
            optimizer.zero_grad()
            b_ind = indices[step*B : (step+1)*B]
            frames = data[b_ind] ## B * T * 64 * 64
            tj_b = tjs[b_ind]
            if CUDA:
                with torch.cuda.device(device):
                    frames = frames.cuda()
                    tj_b = tj_b.cuda()
                    tj_std = tj_std.cuda()

            metrics = objective(models, frames, tj_b, tj_std, mcmc_steps, mnist_mean, crop)
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
        metrics_print = ",  ".join(['%s: %.3f' % (k, v/NUM_BATCHES) for k, v in Metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        time_end = time.time()
        print("(%ds) " % (time_end - time_start) + metrics_print, file=flog)
        flog.close()
        print("Completed  in (%ds),  " % (time_end - time_start) + metrics_print)
