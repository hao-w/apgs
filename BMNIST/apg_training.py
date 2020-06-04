import torch
import time
import numpy as np
from random import shuffle
from BMNIST.apg_modeling import save_model
from BMNIST.apg_objective import apg_objective
# from snr import *

def train(optimizer, model, AT, resampler, apg_sweeps, data_paths, mnist_mean_path, K, num_epochs, sample_size, batch_size, CUDA, DEVICE, MODEL_VERSION):
    """
    ==========
    training function of apg samplers
    ==========
    """
    loss_required = True
    ess_required = True
    mode_required = False
    density_required = True

    print('Start to train the APG samplers with %d groups of datasets...' % len(data_paths))
    mnist_mean = torch.from_numpy(np.load(mnist_mean_path)).float().repeat(sample_size, batch_size, K, 1, 1)

    # iteration = 0
    # fst_mmt = None
    # sec_mmt = None

    for epoch in range(num_epochs):
        shuffle(data_paths)
        for group, data_path in enumerate(data_paths):
            time_start = time.time()
            metrics = dict()
            data = torch.from_numpy(np.load(data_path)).float()
            num_seqs = data.shape[0]
            num_batches = int(num_seqs / batch_size)
            seq_indices = torch.randperm(num_seqs)
            for b in range(num_batches):
                # iteration += 1
                optimizer.zero_grad()
                batch_index = seq_indices[b*batch_size : (b+1)*batch_size]
                frames = data[batch_index].repeat(sample_size, 1, 1, 1, 1)
                if CUDA:
                    with torch.cuda.device(DEVICE):
                        frames = frames.cuda()
                        mnist_mean = mnist_mean.cuda()
                trace = apg_objective(model=model,
                                      AT=AT,
                                      resampler=resampler,
                                      apg_sweeps=apg_sweeps,
                                      frames=frames,
                                      mnist_mean=mnist_mean,
                                      K=K,
                                      loss_required=loss_required,
                                      ess_required=ess_required,
                                      mode_required=mode_required,
                                      density_required=density_required)

                loss_phi = trace['loss_phi'].sum()
                loss_theta = trace['loss_theta'].sum()
                loss_phi.backward(retain_graph=True)
                loss_theta.backward()
                # ## compute the EMA of SNR and variance of gradient estimations
                # fst_mmt, sec_mmt = mmt_grad(model, iteration=iteration, fst_mmt_old=fst_mmt, sec_mmt_old=sec_mmt)
                # if iteration % 1000 == 0:
                #     snr, var = snr_grad(fst_mmt, sec_mmt)
                #     converge_file = open('../results/apg/converge-' + MODEL_VERSION + '.txt', 'a+')
                #     print("snr: %.6f, var: %.6f" % (snr.item(), var.item()), file=converge_file)
                #     converge_file.close()
                optimizer.step()
                if loss_required:
                    assert trace['loss_phi'].shape == (1+apg_sweeps, ), 'ERROR! loss_phi has unexpected shape.'
                    assert trace['loss_theta'].shape == (1+apg_sweeps, ), 'ERROR! loss_theta has unexpected shape.'
                    if 'loss_phi' in metrics:
                        metrics['loss_phi'] += trace['loss_phi'][-1].item()
                    else:
                        metrics['loss_phi'] = trace['loss_phi'][-1].item()
                    if 'loss_theta' in metrics:
                        metrics['loss_theta'] += trace['loss_theta'][-1].item()
                    else:
                        metrics['loss_theta'] = trace['loss_theta'][-1].item()

                if ess_required:
                    assert trace['ess_rws'][0].shape == (batch_size, ), 'ERROR! ess_rws has unexpected shape.'
#                     assert trace['ess_what'].shape == (apg_sweeps, batch_size), 'ERROR! ess_what has unexpected shape.'
#                     assert trace['ess_where'].shape == (apg_sweeps, batch_size), 'ERROR! ess_where has unexpected shape.'
                    if 'ess_rws' in metrics:
                        metrics['ess_rws'] += trace['ess_rws'][0].mean().item()
                    else:
                        metrics['ess_rws'] = trace['ess_rws'][0].mean().item()

#                     if 'ess_what' in metrics:
#                         metrics['ess_what'] += trace['ess_what'].mean(-1)[-1].item()
#                     else:
#                         metrics['ess_what'] = trace['ess_what'].mean(-1)[-1].item()

#                     if 'ess_where' in metrics:
#                         metrics['ess_where'] += trace['ess_where'].mean(-1)[-1].item()
#                     else:
#                         metrics['ess_where'] = trace['ess_where'].mean(-1)[-1].item()
                if density_required:
                    assert trace['density'].shape == (1+apg_sweeps, batch_size), 'ERROR! density has unexpected shape.'
                    if 'density' in metrics:
                        metrics['density'] += trace['density'].mean(-1)[-1].item()
                    else:
                        metrics['density'] = trace['density'].mean(-1)[-1].item()

            save_model(model=model, SAVE_VERSION=MODEL_VERSION)
            metrics_print = ",  ".join(['%s: %.6f' % (k, v/num_batches) for k, v in metrics.items()])
            log_file = open('../results/log-' + MODEL_VERSION + '.txt', 'a+')
            time_end = time.time()
            print("(%ds) Epoch=%d, Group=%d, " % (time_end - time_start, epoch, group) + metrics_print, file=log_file)
            log_file.close()
            print("Epoch=%d, Group=%d completed in (%ds),  " % (epoch, group, time_end - time_start))
