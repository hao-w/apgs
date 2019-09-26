import torch
import time

def train(models, objective, optimizer, data, mcmc_steps, Train_Params):
    (NUM_EPOCHS, T, K, D, S, B, CUDA, device, path) = Train_Params
    """
    training function
    """
    NUM_DATASETS = data.shape[0]
    NUM_BATCHES = int((NUM_DATASETS / B))
    annealed_coefficient = (torch.arange(mcmc_steps+1) + 1).float() / (mcmc_steps+1)
    if CUDA:
        with torch.cuda.device(device):
            annealed_coefficient = annealed_coefficient.cuda()
    for epoch in range(NUM_EPOCHS):
        Metrics = dict()
        time_start = time.time()
        indices = torch.randperm(NUM_DATASETS)
        for step in range(NUM_BATCHES):
            optimizer.zero_grad()
            b_ind = indices[step*B : (step+1)*B]
            ob = data[b_ind] ## B * T * 64 * 64
            if CUDA:
                with torch.cuda.device(device):
                    ob = ob.cuda()
            metrics = objective(models, optimizer, ob, mcmc_steps, K)
            phi_loss = torch.cat(metrics['phi_loss'], 0).sum()
            theta_loss = (torch.cat(metrics['theta_loss'], 0) * annealed_coefficient).sum()
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
