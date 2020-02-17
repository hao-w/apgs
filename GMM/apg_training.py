import sys
sys.path.append('../')
import torch
import time
from utils import shuffler
from apg_modeling import save_model
from apg_objective import apg_objective
from kls_gmm import kl_gmm, kl_gmm_training

def train(optimizer, model, block, resampler, apg_sweeps, data, true_z, num_epochs, sample_size, batch_size, CUDA, DEVICE, MODEL_VERSION):
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

    num_datasets, N, D = data.shape
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
            if CUDA:
                ob = concat_var[:,:,:,:2].cuda().to(DEVICE)
                true_zb = concat_var[:,:,:,2:].cuda().to(DEVICE)
            trace = apg_objective(model=model,
                                  block=block,
                                  resampler=resampler,
                                  apg_sweeps=apg_sweeps,
                                  ob=ob,
                                  loss_required=loss_required,
                                  ess_required=ess_required,
                                  mode_required=mode_required,
                                  density_required=density_required)
            loss = trace['loss'].sum()
            ## gradient step
            loss.backward()
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
                if 'ess' in metrics:
                    metrics['ess'] += trace['ess'].mean(-1)[-1].item()
                else:
                    metrics['ess'] = trace['ess'].mean(-1)[-1].item()
            if density_required:
                assert trace['density'].shape == (1+apg_sweeps, batch_size), 'ERROR! density has unexpected shape.'
                if 'density' in metrics:
                    metrics['density'] += trace['density'].mean(-1)[-1].item()
                else:
                    metrics['density'] = trace['density'].mean(-1)[-1].item()
        save_model(model, MODEL_VERSION)
        metrics_print = ",  ".join(['%s: %.6f' % (k, v/num_batches) for k, v in metrics.items()])
        log_file = open('../results/log-' + MODEL_VERSION + '.txt', 'a+')
        time_end = time.time()
        print(metrics_print + ", KL=%.6f" % (accu_kls / num_batches), file=log_file)
        log_file.close()
        print("Epoch=%d completed  in (%ds),  " % (epoch, time_end - time_start))
