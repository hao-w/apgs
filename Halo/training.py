import torch
import time
from utils import *
from ag_ep_pcg_dec import *

def test(models, objective, Data, Model_Params, Train_Params):
    (NUM_EPOCHS, NUM_DATASETS, S, B, CUDA, device, path) = Train_Params
    SubTrain_Params = (device, S, B) + Model_Params
    ##(N, K, DCharles Levine, mcmc_size) = Model_Params
    indices = torch.randperm(NUM_DATASETS)
    batch_indices = indices[0*B : (0+1)*B]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(S, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    loss, metric_step, reused = objective(models, obs, SubTrain_Params)
    return obs, metric_step, reused

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
