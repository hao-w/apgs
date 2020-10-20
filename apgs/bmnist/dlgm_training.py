import sys
sys.path.append('../')
import torch
import time
import numpy as np
from random import shuffle
from dlgm_modeling import save_model
from dlgm_objective import dlgm_objective

def train(optimizer_phi, optimizer_theta, model, HMC, AT, data_paths, mnist_mean_path, K, num_epochs, sample_size, batch_size, CUDA, DEVICE, MODEL_VERSION):
    """
    ==========
    training function of dlgm
    ==========
    """
    loss_required = True
    ess_required = True
    mode_required = False
    density_required = True
    epsilon = 1e-1
    hmc_num_steps = 10
    print('Start to train the DLGM model with %d groups of datasets...' % len(data_paths))
    mnist_mean = torch.from_numpy(np.load(mnist_mean_path)).float().repeat(sample_size, batch_size, K, 1, 1)

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
                optimizer_phi.zero_grad()
                optimizer_theta.zero_grad()

                batch_index = seq_indices[b*batch_size : (b+1)*batch_size]
                frames = data[batch_index].repeat(sample_size, 1, 1, 1, 1)
                if CUDA:
                    with torch.cuda.device(DEVICE):
                        frames = frames.cuda()
                        mnist_mean = mnist_mean.cuda()
                trace, epsilon = dlgm_objective(model=model,
                                                  HMC=HMC,
                                                  AT=AT,
                                                  frames=frames,
                                                  mnist_mean=mnist_mean,
                                                  K=K,
                                                  epsilon=epsilon,
                                                  hmc_num_steps=hmc_num_steps,
                                                  loss_required=loss_required,
                                                  ess_required=ess_required,
                                                  mode_required=mode_required,
                                                  density_required=density_required)

                loss_phi = trace['loss_phi']
                loss_theta = trace['loss_theta']
                loss_phi.backward(retain_graph=True)
                optimizer_phi.step()
                optimizer_theta.zero_grad()
                loss_theta.backward()
                optimizer_theta.step()
                if loss_required:
                    if 'loss_phi' in metrics:
                        metrics['loss_phi'] += trace['loss_phi'].item()
                    else:
                        metrics['loss_phi'] = trace['loss_phi'].item()
                    if 'loss_theta' in metrics:
                        metrics['loss_theta'] += trace['loss_theta'].item()
                    else:
                        metrics['loss_theta'] = trace['loss_theta'].item()
                if ess_required:
                    assert trace['ess_vae'][0].shape == (batch_size, ), 'ERROR! ess_rws has unexpected shape.'
                    if 'ess_vae' in metrics:
                        metrics['ess_vae'] += trace['ess_vae'][0].mean().item()
                    else:
                        metrics['ess_vae'] = trace['ess_vae'][0].mean().item()

                if density_required:
                    assert trace['density'].shape == (1, batch_size), 'ERROR! density has unexpected shape.'
                    if 'density' in metrics:
                        metrics['density'] += trace['density'].mean(-1)[-1].item()
                    else:
                        metrics['density'] = trace['density'].mean(-1)[-1].item()

            save_model(model=model, SAVE_VERSION=MODEL_VERSION)
            metrics_print = ",  ".join(['%s: %.6f' % (k, v/num_batches) for k, v in metrics.items()])
            log_file = open('../results/vae/log-' + MODEL_VERSION + '.txt', 'a+')
            time_end = time.time()
            print("(%ds) Epoch=%d, Group=%d, HMC rate=%.4f" % (time_end - time_start, epoch, group, HMC.smallest_accept_ratio) + metrics_print, file=log_file)
            log_file.close()
            print("Epoch=%d, Group=%d completed in (%ds),  " % (epoch, group, time_end - time_start))
