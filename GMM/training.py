import torch
import time
from utils import *

def train(models, objective, optimizer, data, Model_Params, Train_Params):
    """
    generic training function
    """
    (NUM_EPOCHS, NUM_DATASETS, S, B, CUDA, device, path) = Train_Params
    SubTrain_Params = (device, S, B) + Model_Params

    NUM_BATCHES = int((NUM_DATASETS / B))
    EPS = torch.FloatTensor([1e-15]).log() ## EPS for KL between categorial distributions
    if CUDA:
        EPS = EPS.cuda().to(device) ## EPS for KL between categorial distributions
    for epoch in range(NUM_EPOCHS):
        metrics = dict()
        time_start = time.time()
        indices = torch.randperm(NUM_DATASETS)
        for step in range(NUM_BATCHES):
            optimizer.zero_grad()
            batch_indices = indices[step*B : (step+1)*B]
            obs = data[batch_indices]
            obs = shuffler(obs).repeat(S, 1, 1, 1)
            if CUDA:
                obs =obs.cuda().to(device)
            loss, metric_step, reused = objective(models, obs, SubTrain_Params)
            ## gradient step
            loss.backward()
            optimizer.step()
            for key in metric_step.keys():
                if key in metrics:
                    metrics[key] += metric_step[key].mean().item()
                else:
                    metrics[key] = metric_step[key].mean().item()
            ## compute KL
            kl_step = kl_train((enc_eta, enc_z), obs, reused, EPS)
            for key in kl_step.keys():
                if key in metrics:
                    metrics[key] += kl_step[key]
                else:
                    metrics[key] = kl_step[key]
        time_end = time.time()
        metrics_print = ",  ".join(['%s: %.3f' % (k, v/NUM_BATCHES) for k, v in metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        print(metrics_print, file=flog)
        flog.close()
        print("epoch: %d\\%d (%ds),  " % (epoch, NUM_EPOCHS, time_end - time_start) + metrics_print)

def test(models, objective, Data, Model_Params, Train_Params):
    (NUM_EPOCHS, NUM_DATASETS, S, B, CUDA, device, path) = Train_Params
    SubTrain_Params = (device, S, B) + Model_Params
    ##(N, K, D, mcmc_size) = Model_Params
    indices = torch.randperm(NUM_DATASETS)
    batch_indices = indices[0*B : (0+1)*B]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(S, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    loss, metric_step, reused = objective(models, obs, SubTrain_Params)
    return obs, metric_step, reused

def kl_train(models, obs, reused, EPS):
    (enc_eta, enc_z) = models
    (state) = reused
    S, B, N, D = obs.shape
    q_eta, p_eta, q_nu = enc_eta(obs, state, K, D)
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value
    q_z, p_z = enc_z.forward(obs, obs_tau, obs_mu, N, K, sample_size, batch_size)
    ## KLs for mu and sigma based on Normal-Gamma prior
    q_alpha = q_eta['precisions'].dist.concentration
    q_beta = q_eta['precisions'].dist.rate
    q_mu = q_eta['means'].dist.loc
    q_pi = q_z['zs'].dist.probs
    pr_alpha = p_eta['precisions'].dist.concentration
    pr_beta = p_eta['precisions'].dist.rate
    pr_mu = p_eta['means'].dist.loc
    pr_pi = p_z['zs'].dist.probs

    _, _, _, K = state.shape
    post_alpha, post_beta, post_mu, post_nu = Post_eta(obs, state, pr_alpha, pr_beta, pr_mu, pr_nu, K, D)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    ## KLs for cluster assignments
    post_logits = Post_z(obs, obs_tau, obs_mu, pr_pi, N, K)
    kl_z_ex, kl_z_in = kls_cats(q_pi.log(), post_logits, EPS)
    kl_step = {"kl_eta_ex" : kl_eta_ex.sum(-1).mean().item(),"kl_eta_in" : kl_eta_in.sum(-1).mean().item(),"kl_z_ex" : kl_z_ex.sum(-1).mean().item(),"kl_z_in" : kl_z_in.sum(-1).mean().item()}
    return kl_step
