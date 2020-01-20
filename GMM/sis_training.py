import sys
sys.path.append('../')
import torch
import time
from utils import shuffler
from apg_modeling import save_model
from sis_objective import sis_objective
from snr import *
from kls_gmm import kl_gmm_training

def train(optimizer, model, apg_sweeps, data, assignemnt_true, num_epochs, sample_size, batch_size, CUDA, DEVICE, MODEL_VERSION):
    """
    ==========
    training function for sis framework
    ==========
    """
    loss_required = True
    ess_required = True
    mode_required = False
    density_required = False
    kl_required = False


    num_datasets = data.shape[0]
    N = data.shape[1]
    num_batches = int((num_datasets / batch_size))
    
    (_, _, enc_apg_eta, generative) = model

    iteration = 0
    fst_mmt = None
    sec_mmt = None
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = dict()
        indices = torch.randperm(num_datasets)
        for b in range(num_batches):
            iteration += 1
            optimizer.zero_grad()
            batch_indices = indices[b*batch_size : (b+1)*batch_size]
            concat_var = torch.cat((data[batch_indices], assignemnt_true[batch_indices]), -1)
            concat_var = shuffler(concat_var).repeat(sample_size, 1, 1, 1)
            if CUDA:
                ob = concat_var[:,:,:,:2].cuda().to(DEVICE)
                z_true = concat_var[:,:,:,2:].cuda().to(DEVICE)
            trace = sis_objective(model=model,
                                  apg_sweeps=apg_sweeps,
                                  ob=ob,
                                  loss_required=loss_required,
                                  ess_required=ess_required,
                                  mode_required=mode_required,
                                  density_required=density_required,
                                  kl_required=kl_required)
            loss = trace['loss'].sum()
            ## gradient step
            loss.backward()
            ## compute the EMA of SNR and variance of gradient estimations
            fst_mmt, sec_mmt = mmt_grad(model, iteration=iteration, fst_mmt_old=fst_mmt, sec_mmt_old=sec_mmt)
            if iteration % 200 == 0:
                snr, var = snr_grad(fst_mmt, sec_mmt)
                inckl_eta_truez = kl_gmm_training(enc_apg_eta, generative, ob, z_true)
                converge_file = open('../results/sis_new/converge-' + MODEL_VERSION + '.txt', 'a+')
                print("snr: %.6f, var: %.6f, inckl_gen:%.6f" % (snr.item(), var.item(), inckl_eta_truez.item()), file=converge_file)
                converge_file.close()
            optimizer.step()
            if loss_required:
                assert trace['loss'].shape == (1, ), 'ERROR! loss has unexpected shape.'
                if 'loss' in metrics:
                    metrics['loss'] += trace['loss'][-1].item()
                else:
                    metrics['loss'] = trace['loss'][-1].item()
            if ess_required:
                assert trace['ess_sis'].shape == (1+apg_sweeps*2, batch_size), 'ERROR! ess_sis has unexpected shape.'
                if 'ess_sis' in metrics:
                    metrics['ess_sis'] += trace['ess_sis'][0].mean().item()
                else:
                    metrics['ess_sis'] = trace['ess_sis'][0].mean().item()

            if density_required:
                assert trace['density'].shape == (1+apg_sweeps, batch_size), 'ERROR! density has unexpected shape.'
                if 'density' in metrics:
                    metrics['density'] += trace['density'].mean(-1)[-1].item()
                else:
                    metrics['density'] = trace['density'].mean(-1)[-1].item()
            if kl_required:
                assert trace['inckl_eta'].shape == (batch_size,), 'ERROR! inckl_eta has unexpected shape.'
                assert trace['inckl_z'].shape == (batch_size,), 'ERROR! inckl_z has unexpected shape.'
                if 'inckl_eta' in metrics:
                    metrics['inckl_eta'] += trace['inckl_eta'].mean().item()
                else:
                    metrics['inckl_eta'] = trace['inckl_eta'].mean().item()

                if 'inckl_z' in metrics:
                    metrics['inckl_z'] += trace['inckl_z'].mean().item()
                else:
                    metrics['inckl_z'] = trace['inckl_z'].mean().item()
        save_model(model, MODEL_VERSION)
        metrics_print = ",  ".join(['%s: %.6f' % (k, v/num_batches) for k, v in metrics.items()])
        log_file = open('../results/sis_new/log-' + MODEL_VERSION + '.txt', 'a+')
        time_end = time.time()
        print("(%ds) " % (time_end - time_start) + metrics_print, file=log_file)
        log_file.close()
        print("Epoch=%d completed  in (%ds),  " % (epoch, time_end - time_start))
