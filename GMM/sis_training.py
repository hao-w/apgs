import sys
sys.path.append('../')
import torch
import time
from utils import shuffler
from apg_modeling import save_model
from sis_objective import sis_objective
from kls_gmm import kl_gmm, kl_gmm_training

def train(optimizer, model, apg_sweeps, data, true_z, num_epochs, sample_size, batch_size, CUDA, DEVICE, MODEL_VERSION):
    """
    ==========
    training function for sis framework
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

    (_, _, enc_apg_eta, generative) = model

    for epoch in range(num_epochs):
        accu_kls = 0.0
        time_start = time.time()
        metrics = dict()
        indices = torch.randperm(num_datasets)
        for b in range(num_batches):
            optimizer.zero_grad()
            batch_indices = indices[b*batch_size : (b+1)*batch_size]
            concat_var = torch.cat((data[batch_indices], true_z[batch_indices]), -1)
            concat_var = shuffler(concat_var).repeat(sample_size, 1, 1, 1)
            # ob = shuffler(data[batch_indices]).repeat(sample_size, 1, 1, 1)
            if CUDA:
                ob = concat_var[:,:,:,:2].cuda().to(DEVICE)
                true_zb = concat_var[:,:,:,2:].cuda().to(DEVICE)
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
            optimizer.step()
            inckl_eta_truez = kl_gmm_training(enc_apg_eta, generative, ob, z=true_zb)
            accu_kls += inckl_eta_truez.item()
            # if iteration % 200 == 0:
            #     wall_clock_end = time.time()
            #     converge_file = open('../results/sis_convergences/' + MODEL_VERSION + '.txt', 'a+')
            #     print("time:%d, inckl:%.6f" % (wall_clock_end - wall_clock_start, accu_kls/ 200), file=converge_file)
            #     converge_file.close()
            #     accu_kls = 0.0

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
        log_file = open('../results/trainkl_apg/log-' + MODEL_VERSION + '.txt', 'a+')
        time_end = time.time()
        print("TrueKL=%.6f, " % (accu_kls / num_batches) + metrics_print, file=log_file)
        log_file.close()
        print("Epoch=%d completed  in (%ds),  " % (epoch, time_end - time_start))
