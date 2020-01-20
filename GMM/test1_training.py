import sys
sys.path.append('../')
import torch
import time
from utils import shuffler
from apg_modeling import save_model
from test1_objective import test1_objective

def train(optimizer, model, resampler, apg_sweeps, data, num_epochs, sample_size, batch_size, CUDA, DEVICE, MODEL_VERSION):
    """
    ==========
    training function for apg samplers
    ==========
    """
    loss_required = True
    ess_required = True
    mode_required = False
    density_required = True
    kl_required = True

    num_datasets = data.shape[0]
    N = data.shape[1]
    num_batches = int((num_datasets / batch_size))
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = dict()
        indices = torch.randperm(num_datasets)
        for b in range(num_batches):
            optimizer.zero_grad()
            batch_indices = indices[b*batch_size : (b+1)*batch_size]
            ob = shuffler(data[batch_indices]).repeat(sample_size, 1, 1, 1)
            if CUDA:
                ob = ob.cuda().to(DEVICE)
            trace = test1_objective(model=model,
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
            optimizer.step()
            if loss_required:
                assert trace['loss'].shape == (1+apg_sweeps, ), 'ERROR! loss has unexpected shape.'
                if 'loss' in metrics:
                    metrics['loss'] += trace['loss'][-1].item()
                else:
                    metrics['loss'] = trace['loss'][-1].item()
            if ess_required:
                assert trace['ess_rws'].shape == (1+apg_sweeps, batch_size), 'ERROR! ess_rws has unexpected shape.'
                # assert trace['ess_eta'].shape == (apg_sweeps, batch_size), 'ERROR! ess_eta has unexpected shape.'
                # assert trace['ess_z'].shape == (apg_sweeps, batch_size), 'ERROR! ess_z has unexpected shape.'
                if 'ess_rws' in metrics:
                    metrics['ess_rws'] += trace['ess_rws'].mean(-1)[-1].item()
                else:
                    metrics['ess_rws'] = trace['ess_rws'].mean(-1)[-1].item()

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
        log_file = open('../results/log-' + MODEL_VERSION + '.txt', 'a+')
        time_end = time.time()
        print("(%ds) " % (time_end - time_start) + metrics_print, file=log_file)
        log_file.close()
        print("Epoch=%d completed  in (%ds),  " % (epoch, time_end - time_start))
