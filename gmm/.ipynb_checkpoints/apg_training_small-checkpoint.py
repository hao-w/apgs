import sys
sys.path.append('../')
import torch
import time
from utils import shuffler
from apg_modeling import save_model
from apg_objective_small import apg_objective
from snr import *
from kls_gmm import kl_gmm, kl_gmm_training

def train(optimizer, model, resampler, apg_sweeps, data, true_z, num_epochs, sample_size, batch_size, CUDA, DEVICE, MODEL_VERSION):
    """
    ==========
    training function for apg samplers
    ==========
    """
    loss_required = True
    ess_required = True
    mode_required = False
    density_required = False
    kl_required = True

    num_datasets = data.shape[0]
    N = data.shape[1]
    num_batches = int((num_datasets / batch_size))

    iteration = 0

    fst_mmt = None
    sec_mmt = None
    (_, _, enc_apg_eta, generative) = model
    for epoch in range(num_epochs):
        accu_kls = 0.0
        time_start = time.time()
        metrics = dict()
        indices = torch.randperm(num_datasets)
        for b in range(num_batches):
            iteration += 1
            optimizer.zero_grad()
            batch_indices = indices[b*batch_size : (b+1)*batch_size]
            concat_var = torch.cat((data[batch_indices], true_z[batch_indices]), -1)
            concat_var = shuffler(concat_var).repeat(sample_size, 1, 1, 1)
            # ob = shuffler(data[batch_indices]).repeat(sample_size, 1, 1, 1)
            if CUDA:
                ob = concat_var[:,:,:,:2].cuda().to(DEVICE)
                true_zb = concat_var[:,:,:,2:].cuda().to(DEVICE)
                # ob = ob.cuda().to(DEVICE)
            trace = apg_objective(model=model,
                                  resampler=resampler,
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
            if iteration % 1000 == 0:
                snr, var = snr_grad(fst_mmt, sec_mmt)
                converge_file = open('../results/sis/converge-' + MODEL_VERSION + '.txt', 'a+')
                print("snr: %.6f, var: %.6f" % (snr.item(), var.item()), file=converge_file)
                converge_file.close()
            optimizer.step()
            inckl_eta_truez = kl_gmm_training(enc_apg_eta, generative, ob, z=true_zb)
            accu_kls += inckl_eta_truez.item()

            if loss_required:
                assert trace['loss'].shape == (1+apg_sweeps, ), 'ERROR! loss has unexpected shape.'
                if 'loss' in metrics:
                    metrics['loss'] += trace['loss'][-1].item()
                else:
                    metrics['loss'] = trace['loss'][-1].item()
            if ess_required:
                # if 'ess_rws' in metrics:
                #     metrics['ess_rws'] += trace['ess_rws'].mean(-1)[-1].item()
                # else:
                #     metrics['ess_rws'] = trace['ess_rws'].mean(-1)[-1].item()
                # assert trace['ess_rws'][0].shape == (batch_size, ), 'ERROR! ess_rws has unexpected shape.'
                # assert trace['ess_eta'].shape == (apg_sweeps, batch_size), 'ERROR! ess_eta has unexpected shape.'
                # assert trace['ess_z'].shape == (apg_sweeps, batch_size), 'ERROR! ess_z has unexpected shape.'
                if 'ess_rws' in metrics:
                    metrics['ess_rws'] += trace['ess_rws'][0].mean().item()
                else:
                    metrics['ess_rws'] = trace['ess_rws'][0].mean().item()

                if 'ess_eta' in metrics:
                    metrics['ess_eta'] += trace['ess_eta'].mean(-1)[-1].item()
                else:
                    metrics['ess_eta'] = trace['ess_eta'].mean(-1)[-1].item()

                if 'ess_z' in metrics:
                    metrics['ess_z'] += trace['ess_z'].mean(-1)[-1].item()
                else:
                    metrics['ess_z'] = trace['ess_z'].mean(-1)[-1].item()
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
        log_file = open('../results/blocks_apg/log-' + MODEL_VERSION + '.txt', 'a+')
        time_end = time.time()
        print("TrueKL=%.6f, " % (accu_kls / num_batches) + metrics_print, file=log_file)
        log_file.close()
        print("Epoch=%d completed  in (%ds),  " % (epoch, time_end - time_start))
