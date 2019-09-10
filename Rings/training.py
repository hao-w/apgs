import torch
import time
from utils import *
from ag_ep_dec import *
# from os_ep_dec import *

def train_10size(models, optimizer, data, Model_Params, Train_Params):
    """
    training function for datasets with various sizes
    """
    (NUM_EPOCHS, S, B, CUDA, device, path) = Train_Params
    (K, D, mcmc_size) = Model_Params
    GROUP_SIZE = len(data)
    NUM_DATASETS = data[0].shape[0]
    NUM_BATCHES = int((NUM_DATASETS / B))
    TOTAL_BATCHES = NUM_BATCHES * GROUP_SIZE
    for epoch in range(NUM_EPOCHS):
        time_start = time.time()
        ELBO = 0.0
        EUBO = 0.0
        ESS = 0.0
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
                eubo, elbo, theta_loss, ess = AG_pcg(models, ob, K, mcmc_size, device)
                eubo.backward(retain_graph=True)
                theta_loss.backward()
                optimizer.step()
                ELBO += elbo.detach()
                EUBO += eubo.detach()
                ESS += ess

        flog = open('../results/log-' + path, 'a+')
        print('epoch=%d, eubo=%.4f, elbo=%.4f, ess=%.4f' % (epoch,  EUBO / TOTAL_BATCHES, ELBO / TOTAL_BATCHES, ESS / TOTAL_BATCHES), file=flog)
        flog.close()
        time_end = time.time()
        print('epoch=%d, eubo=%.4f, elbo=%.4f, ess=%.4f (%ds)' % (epoch,  EUBO / TOTAL_BATCHES, ELBO / TOTAL_BATCHES, ESS / TOTAL_BATCHES, time_end - time_start))

def train(models, optimizer, OB, num_epochs, mcmc_size, S, B, K, CUDA, device, PATH):
    NUM_DATASETS = OB.shape[0]
    NUM_BATCHES = int((NUM_DATASETS / B))
    for epoch in range(num_epochs):
        time_start = time.time()
        ELBO = 0.0
        EUBO = 0.0
        ESS = 0.0
        indices = torch.randperm(NUM_DATASETS)
        for step in range(NUM_BATCHES):
            optimizer.zero_grad()
            batch_indices = indices[step*B : (step+1)*B]
            ob = OB[batch_indices]
            ob = shuffler(ob).repeat(S, 1, 1, 1)
            if CUDA:
                with torch.cuda.device(device):
                    ob = ob.cuda()
            eubo, elbo, theta_loss, ess = AG_pcg(models, ob, K, mcmc_size, device)
            eubo.backward(retain_graph=True)
            theta_loss.backward()
            optimizer.step()
            ELBO += elbo.detach()
            EUBO += eubo.detach()
            ESS += ess

        flog = open('../results/log-' + PATH, 'a+')
        print('epoch=%d, eubo=%.4f, elbo=%.4f, ess=%.4f' % (epoch,  EUBO / NUM_BATCHES, ELBO / NUM_BATCHES, ESS / NUM_BATCHES), file=flog)
        flog.close()
        time_end = time.time()
        print('epoch=%d, eubo=%.4f, elbo=%.4f, ess=%.4f (%ds)' % (epoch,  EUBO / NUM_BATCHES, ELBO / NUM_BATCHES, ESS / NUM_BATCHES, time_end - time_start))
#
# def train_os(models, optimizer, OB, num_epochs, S, B, K, CUDA, device, PATH):
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
#             eubo, elbo, theta_loss, ess = OS_pcg(models, ob, K, device)
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
