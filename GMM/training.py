import torch
import time
from utils import *

def train(objective, optimizer, data, Train_Params, Model_Params, models):
    """
    generic training function
    """
    (NUM_EPOCHS, NUM_BATCHES, NUM_SEQS, CUDA, device, path) = Train_Params
    (N, K, D, mcmc_size, sample_size, batch_size) = Model_Params
    for epoch in range(NUM_EPOCHS):
        metrics = dict()
        time_start = time.time()
        indices = torch.randperm(NUM_SEQS)
        for step in range(NUM_BATCHES):
            optimizer.zero_grad()
            batch_indices = indices[step*batch_size : (step+1)*batch_size]
            obs = data[batch_indices]
            obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
            if CUDA:
                obs =obs.cuda().to(device)
            loss, metric_step, reused = objective(obs, Model_Params, device, models)
            ## gradient step
            loss.backward()
            optimizer.step()
            for key in metric_step.keys():
                if key in metrics:
                    metrics[key] += metric_step[key]
                else:
                    metrics[key] = metric_step[key]
            ## compute KL
            kl_step = kl_train(obs, K, reused)
            for key in kl_step.keys():
                if key in metrics:
                    metrics[key] += kl_step[key]
                else:
                    metrics[key] = kl_step[key]
        time_end = time.time()
        metrics_print = ",  ".join(['%s: %.3f' % (k, float(v)/NUM_BATCHES) for k, v in metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        print(metrics_print, file=flog)
        flog.close()
        print("epoch: %d\\%d (%ds),  " % (epoch, NUM_EPOCHS, time_end - time_start) + metrics_print)

def test(objective, Data, Train_Params, Model_Params, models):
    (NUM_EPOCHS, NUM_BATCHES, NUM_SEQS, CUDA, device, path) = Train_Params
    (N, K, D, mcmc_size, sample_size, batch_size) = Model_Params
    indices = torch.randperm(NUM_SEQS)
    batch_indices = indices[0*batch_size : (0+1)*batch_size]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    loss, metric_step, reused = objective(obs, Model_Params, device, models)
    return obs, metric_step, reused

def kl_train(obs, K, reused):
    (q_eta, p_eta, q_z, p_z, q_nu, pr_nu) = reused
    _, _, N, D = obs.shape
    ## KLs for mu and sigma based on Normal-Gamma prior
    q_alpha = q_eta['precisions'].dist.concentration
    q_beta = q_eta['precisions'].dist.rate
    q_mu = q_eta['means'].dist.loc
    q_pi = q_z['zs'].dist.probs

    pr_alpha = p_eta['precisions'].dist.concentration
    pr_beta = p_eta['precisions'].dist.rate
    pr_mu = p_eta['means'].dist.loc
    pr_pi = p_z['zs'].dist.probs

    states = q_z['zs'].value
    obs_mu = q_eta['means'].value
    obs_tau = q_eta['precisions'].value

    post_alpha, post_beta, post_mu, post_nu = Post_eta(obs, states, pr_alpha, pr_beta, pr_mu, pr_nu, K, D)
    kl_eta_ex, kl_eta_in = kls_NGs(q_alpha, q_beta, q_mu, q_nu, post_alpha, post_beta, post_mu, post_nu)
    ## KLs for cluster assignments
    post_logits = Post_z(obs, obs_tau, obs_mu, pr_pi, N, K)
    kl_z_ex, kl_z_in = kls_cats(q_pi.log(), post_logits)
    kl_step = {"kl_eta_ex" : kl_eta_ex.sum(-1).mean().item(),"kl_eta_in" : kl_eta_in.sum(-1).mean().item(),"kl_z_ex" : kl_z_ex.sum(-1).mean().item(),"kl_z_in" : kl_z_in.sum(-1).mean().item()}
    return kl_step



def train_baseline(objective, optimizer, data, Train_Params, Model_Params, models):
    """
    generic training function
    """
    (NUM_EPOCHS, NUM_BATCHES, NUM_SEQS, CUDA, device, path) = Train_Params
    (N, K, D, sample_size, batch_size) = Model_Params
    for epoch in range(NUM_EPOCHS):
        metrics = dict()
        time_start = time.time()
        indices = torch.randperm(NUM_SEQS)
        for step in range(NUM_BATCHES):
            optimizer.zero_grad()
            batch_indices = indices[step*batch_size : (step+1)*batch_size]
            obs = data[batch_indices]
            obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
            if CUDA:
                obs =obs.cuda().to(device)
            loss, metric_step, reused = objective(obs, Model_Params, device, models)
            ## gradient step
            loss.backward()
            optimizer.step()
            for key in metric_step.keys():
                if key in metrics:
                    metrics[key] += metric_step[key]
                else:
                    metrics[key] = metric_step[key]
        time_end = time.time()
        metrics_print = ",  ".join(['%s: %.3f' % (k, float(v)/NUM_BATCHES) for k, v in metrics.items()])
        flog = open('../results/log-' + path + '.txt', 'a+')
        print(metrics_print, file=flog)
        flog.close()
        print("epoch: %d\\%d (%ds),  " % (epoch, NUM_EPOCHS, time_end - time_start) + metrics_print)

def test_baseline(objective, Data, Train_Params, Model_Params, models):
    (NUM_EPOCHS, NUM_BATCHES, NUM_SEQS, CUDA, device, path) = Train_Params
    (N, K, D, sample_size, batch_size) = Model_Params
    indices = torch.randperm(NUM_SEQS)
    batch_indices = indices[0*batch_size : (0+1)*batch_size]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(sample_size, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    loss, metric_step, reused = objective(obs, Model_Params, device, models)
    return obs, metric_step, reused
