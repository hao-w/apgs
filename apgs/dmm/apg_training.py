import os
import torch
import time
from apgs.dmm.models import Enc_rws_mu, Enc_apg_local, Enc_apg_mu, Decoder
from apgs.dmm.objectives import apg_objective

def train(objective, optimizer, models, data, K, num_epochs, sample_size, batch_size, CUDA, device, **kwargs):
    """
    training function of apg samplers
    """
    result_flags = {'loss_required' : True, 'ess_required' : True, 'mode_required' : False, 'density_required': True}
    num_batches = int((data.shape[0] / batch_size))
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = dict()
        data = shuffler(data)
        for b in range(num_batches):
            optimizer.zero_grad()
            x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1)
            if CUDA:
                x = x.cuda().to(device)
            trace = objective(models, x, K, result_flags, **kwargs)
            loss_phi = trace['loss_phi'].sum()
            loss_theta = trace['loss_theta'][-1] * kwargs['num_sweeps']
            loss_phi.backward(retain_graph=True)
            loss_theta.backward()
            optimizer.step()
            if 'loss_phi' in metrics:
                metrics['loss_phi'] += trace['loss_phi'][-1].item()
            else:
                metrics['loss_phi'] = trace['loss_phi'][-1].item()
            if 'loss_theta' in metrics:
                metrics['loss_theta'] += trace['loss_theta'][-1].item()
            else:
                metrics['loss_theta'] = trace['loss_theta'][-1].item()
#             print(trace['ess'].mean(-1))
            if 'ess' in metrics:
                metrics['ess'] += trace['ess'][-1].mean().item()
            else:
                metrics['ess'] = trace['ess'][-1].mean().item()

            if 'density' in metrics:
                metrics['density'] += trace['density'][-1].mean().item()
            else:
                metrics['density'] = trace['density'][-1].mean().item()
        save_apg_models(models, model_version)
        metrics_print = ",  ".join(['%s: %.4f' % (k, v/num_batches) for k, v in metrics.items()])
        if not os.path.exists('results/'):
            os.makedirs('results/')
        log_file = open('results/log-' + model_version + '.txt', 'a+')
        time_end = time.time()
        print(metrics_print, file=log_file)
        log_file.close()
        print("Epoch=%d / %d (%ds),  " % (epoch+1, num_epochs, time_end - time_start))
        
def shuffler(data):
    """
    shuffle the DMM datasets by both permuting the order of GMM instances (w.r.t. DIM1) and permuting the order of data points in each instance (w.r.t. DIM2)
    """
    DIM1, DIM2, DIM3 = data.shape
    indices_DIM1 = torch.randperm(DIM1)
    data = data[indices_DIM1]
    indices_DIM2 = torch.cat([torch.randperm(DIM2).unsqueeze(0) for b in range(DIM1)], dim=0)
    data = torch.gather(data, 1, indices_DIM2.unsqueeze(-1).repeat(1, 1, DIM3))
    return data

def init_apg_models(K, D, num_hidden_mu, num_nss, num_hidden_local, num_hidden_dec, recon_sigma, CUDA, device, load_version=None, lr=None):
    """
    initialization function for APG samplers
    """
    enc_rws_mu = Enc_rws_mu(K, D, num_hidden_mu, num_nss)
    enc_apg_local = Enc_apg_local(K, D, num_hidden_local)
    enc_apg_mu = Enc_apg_mu(K, D, num_hidden_mu, num_nss)
    dec = Decoder(K, D, num_hidden_dec, recon_sigma, CUDA, device)
    if CUDA:
        with torch.cuda.device(device):
            enc_rws_mu.cuda()
            enc_apg_local.cuda()
            enc_apg_mu.cuda()
            dec.cuda()
            
    if load_version is not None:
        weights = torch.load("weights/cp-%s" % load_version)
        enc_rws_mu.load_state_dict(weights['enc-rws-mu'])
        enc_apg_local.load_state_dict(weights['enc-apg-local'])
        enc_apg_mu.load_state_dict(weights['enc-apg-mu'])
        dec.load_state_dict(weights['dec'])
    if lr is not None:
        assert isinstance(lr, float)
        optimizer =  torch.optim.Adam(list(enc_rws_mu.parameters())+list(enc_apg_local.parameters())+list(enc_apg_mu.parameters())+list(dec.parameters()),lr=lr, betas=(0.9, 0.99))
        return (enc_rws_mu, enc_apg_local, enc_apg_mu, dec), optimizer
    else: # testing
        for p in enc_rws_mu.parameters():
            p.requires_grad = False
        for p in enc_apg_local.parameters():
            p.requires_grad = False
        for p in enc_apg_mu.parameters():
            p.requires_grad = False
        for p in dec.parameters():
            p.requires_grad = False
        return (enc_rws_mu, enc_apg_local, enc_apg_mu, dec)

def save_apg_models(models, save_version):
    """
    saving function for APG samplers
    """
    (enc_rws_mu, enc_apg_local, enc_apg_mu, dec) = models
    checkpoint = {
        'enc-rws-mu' : enc_rws_mu.state_dict(),
        'enc-apg-local' : enc_apg_local.state_dict(),
        'enc-apg-mu' : enc_apg_mu.state_dict(),
        'dec' : dec.state_dict()
    }
    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    torch.save(checkpoint, "weights/cp-%s" % save_version)
    
def init_rws_models(K, D, num_hidden_mu, num_nss, num_hidden_local, num_hidden_dec, recon_sigma, CUDA, device, load_version=None, lr=None):
    """
    initialization function for RWS method
    """
    enc_rws_mu = Enc_rws_mu(K, D, num_hidden_mu, num_nss)
    enc_rws_local = Enc_apg_local(K, D, num_hidden_local)
    dec = Decoder(K, D, num_hidden_dec, recon_sigma, CUDA, device)
    if CUDA:
        with torch.cuda.device(device):
            enc_rws_mu.cuda()
            enc_rws_local.cuda()
            dec.cuda()
    if load_version is not None:
        weights = torch.load("weights/cp-%s" % load_version)
        enc_rws_mu.load_state_dict(weights['enc-rws-mu'])
        enc_rws_local.load_state_dict(weights['enc-rws-local'])
        dec.load_state_dict(weights['dec'])
    if lr is not None:
        assert isinstance(lr, float)
        optimizer =  torch.optim.Adam(list(enc_rws_mu.parameters())+list(enc_rws_local.parameters())+list(dec.parameters()),lr=lr, betas=(0.9, 0.99))
        return (enc_rws_mu, enc_rws_local, dec), optimizer
    else: # testing
        for p in enc_rws_mu.parameters():
            p.requires_grad = False
        for p in enc_rws_local.parameters():
            p.requires_grad = False
        return (enc_rws_mu, enc_rws_local, dec)

def save_rws_models(models, save_version):
    """
    saving function for RWS method
    """
    (enc_rws_mu, enc_rws_local, dec) = models
    checkpoint = {
        'enc-rws-mu' : enc_rws_mu.state_dict(),
        'enc-rws-local' : enc_rws_local.state_dict(),
        'dec' : dec.state_dict()
    }
    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    torch.save(checkpoint, "weights/cp-%s" % save_version)

if __name__ == '__main__':
    import torch
    import numpy as np
    import argparse
    from apgs.resampler import Resampler
    from apgs.dmm.objectives import apg_objective, rws_objective
    parser = argparse.ArgumentParser('DMM Clustering Task')
    parser.add_argument('--data_dir', default='../../data/dmm/')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--budget', default=70, type=int)
    parser.add_argument('--num_sweeps', default=7, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--resample_strategy', default='systematic', choices=['systematic', 'multinomial'])
    parser.add_argument('--num_clusters', default=4, type=int)
    parser.add_argument('--data_dim', default=2, type=int)
    parser.add_argument('--num_hidden_mu', default=32, type=int)
    parser.add_argument('--num_nss', default=8, type=int)
    parser.add_argument('--num_hidden_local', default=32, type=int)
    parser.add_argument('--num_hidden_dec', default=32, type=int)
    parser.add_argument('--recon_sigma', default=0.5, type=float)
    args = parser.parse_args()
    sample_size = int(args.budget / args.num_sweeps)
    CUDA = torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.device)

    data = torch.from_numpy(np.load(args.data_dir + 'ob.npy')).float() 
    print('Start training for dmm clustering task..')
    if args.num_sweeps == 1: ## rws method
        model_version = 'rws-dmm-num_samples=%s' % (sample_size)
        print('version='+ model_version)
        models, optimizer = init_rws_models(args.num_clusters, args.data_dim, args.num_hidden_mu, args.num_nss, args.num_hidden_local, args.num_hidden_dec, args.recon_sigma, CUDA, device, load_version=None, lr=args.lr)
        train(rws_objective, optimizer, models, data, args.num_clusters, args.num_epochs, sample_size, args.batch_size, CUDA, device)
        
    elif args.num_sweeps > 1: ## apg sampler
        model_version = 'apg-dmm-num_sweeps=%s-num_samples=%s' % (args.num_sweeps, sample_size)
        print('version=' + model_version)
        models, optimizer = init_apg_models(args.num_clusters, args.data_dim, args.num_hidden_mu, args.num_nss, args.num_hidden_local, args.num_hidden_dec, args.recon_sigma, CUDA, device, load_version=None, lr=args.lr)
        resampler = Resampler(args.resample_strategy, sample_size, CUDA, device)
        train(apg_objective, optimizer, models, data, args.num_clusters, args.num_epochs, sample_size, args.batch_size, CUDA, device, num_sweeps=args.num_sweeps, resampler=resampler)
        
    else:
        raise ValueError
        