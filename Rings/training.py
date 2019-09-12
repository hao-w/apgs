import torch
import time
from utils import *

def train(models, objective, optimizer, data, mcmc_steps, Train_Params):
    """
    training function for datasets with various sizes
    """
    (NUM_EPOCHS, K, D, S, B, CUDA, device, path) = Train_Params
    GROUP_SIZE = len(data)
    NUM_DATASETS = data[0].shape[0]
    NUM_BATCHES = int((NUM_DATASETS / B))
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
                metrics = objective(models, ob, mcmc_steps, K)
                phi_loss = torch.cat(metrics['phi_loss'], 0).sum()
                theta_loss = torch.cat(metrics['theta_loss'], 0).sum()
                phi_loss.backward(retain_graph=True)
                theta_loss.backward()
                optimizer.step()
                for key in metrics.keys():
                    if key in Metrics:
                        Metrics[key] += metrics[key][-1].item()
                    else:
                        Metrics[key] = metrics[key][-1].item()

        metrics_print = ",  ".join(['%s: %.3f' % (k, v/(GROUP_SIZE*NUM_BATCHES)) for k, v in Metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        print(metrics_print, file=flog)
        flog.close()
        time_end = time.time()
        print("Completed  in (%ds),  " % (time_end - time_start) + metrics_print)

# def train(models, optimizer, OB, num_epochs, mcmc_size, S, B, K, CUDA, device, PATH):
#     NUM_DATASETS = OB.shape[0]
#     NUM_BATCHES = int((NUM_DATASETS / B))
#     for epoch in range(num_epochs):
#         time_start = time.time()
#         ELBO = 0.0
#         EUBO = 0.0
#         ESS = 0.0
#         indices = torch.randperm(NUM_DATASETS)
#         for step in range(NUM_BATCHES):
#             optimizer.zero_grad()
#             batch_indices = indices[step*B : (step+1)*B]
#             ob = OB[batch_indices]
#             ob = shuffler(ob).repeat(S, 1, 1, 1)
#             if CUDA:
#                 with torch.cuda.device(device):
#                     ob = ob.cuda()
#             eubo, elbo, theta_loss, ess = AG_pcg(models, ob, K, mcmc_size, device)
#             eubo.backward(retain_graph=True)
#             theta_loss.backward()
#             optimizer.step()
#             ELBO += elbo.detach()
#             EUBO += eubo.detach()
#             ESS += ess
#
#         flog = open('../results/log-' + PATH, 'a+')
#         print('epoch=%d, eubo=%.4f, elbo=%.4f, ess=%.4f' % (epoch,  EUBO / NUM_BATCHES, ELBO / NUM_BATCHES, ESS / NUM_BATCHES), file=flog)
#         flog.close()
#         time_end = time.time()
#         print('epoch=%d, eubo=%.4f, elbo=%.4f, ess=%.4f (%ds)' % (epoch,  EUBO / NUM_BATCHES, ELBO / NUM_BATCHES, ESS / NUM_BATCHES, time_end - time_start))
