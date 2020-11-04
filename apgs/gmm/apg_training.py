import os
import torch
import time
from apgs.gmm.kls_gmm import kls_eta
from apgs.gmm.models import Enc_rws_eta, Enc_apg_eta, Enc_apg_z, Generative

def train(objective, optimizer, models, data, assignments, num_epochs, sample_size, batch_size, CUDA, device, **kwargs):
    """
    training function for apg samplers
    """
    result_flags = {'loss_required' : True, 'ess_required' : True, 'mode_required' : False, 'density_required': True}
    num_batches = int((data.shape[0] / batch_size))
    for epoch in range(num_epochs):
        time_start = time.time()
        metrics = {'ess': 0.0, 'density' : 0.0, 'inc_kl' : 0.0, 'exc_kl' : 0.0}
#         data, assignments = shuffler(data, assignments)
        for b in range(num_batches):
            optimizer.zero_grad()
            if CUDA:
                x = data[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1).cuda().to(device)
                z_true = assignments[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1).cuda().to(device)
            trace = objective(models, x, result_flags, **kwargs)
            loss = trace['loss'].sum()
            loss.backward()
            optimizer.step()
            metrics['ess'] += trace['ess'][-1].mean()
            metrics['density'] += trace['density'][-1].mean() 
            exc_kl, inc_kl = kls_eta(models, x, z_true)
            metrics['inc_kl'] += inc_kl
            metrics['exc_kl'] += exc_kl
        save_apg_models(models, model_version)
        metrics_print = ", ".join(['%s=%.2f' % (k, v / num_batches) for k, v in metrics.items()])
        if not os.path.exists('results/'):
            os.makedirs('results/')
        log_file = open('results/log-' + model_version + '.txt', 'a+')
        time_end = time.time()
        print(metrics_print, file=log_file)
        log_file.close()
        print("Epoch=%d / %d (%ds),  " % (epoch+1, num_epochs, time_end - time_start))
        
def shuffler(data, assignments):
    """
    shuffle the GMM datasets by both permuting the order of GMM instances (w.r.t. DIM1) and permuting the order of data points in each instance (w.r.t. DIM2)
    """
    concat_var = torch.cat((data, assignments), dim=-1)
    DIM1, DIM2, DIM3 = concat_var.shape
    indices_DIM1 = torch.randperm(DIM1)
    concat_var = concat_var[indices_DIM1]
    indices_DIM2 = torch.cat([torch.randperm(DIM2).unsqueeze(0) for b in range(DIM1)], dim=0)
    concat_var = torch.gather(concat_var, 1, indices_DIM2.unsqueeze(-1).repeat(1, 1, DIM3))
    return concat_var[:,:,:2], concat_var[:,:,2:]

def init_apg_models(K, D, num_hidden_z, CUDA, device, load_version=None, lr=None):
    """
    ==========
    initialization function for APG samplers
    ==========
    """
    enc_rws_eta = Enc_rws_eta(K, D)
    enc_apg_z = Enc_apg_z(K, D, num_hidden_z)
    enc_apg_eta = Enc_apg_eta(K, D)
    generative = Generative(K, D, CUDA, device)
    if CUDA:
        with torch.cuda.device(device):
            enc_rws_eta.cuda()
            enc_apg_z.cuda()
            enc_apg_eta.cuda()
    if load_version is not None:
        weights = torch.load("../weights/cp-%s" % load_version)
        enc_rws_eta.load_state_dict(weights['enc-rws-eta'])
        enc_apg_z.load_state_dict(weights['enc-apg-z'])
        enc_apg_eta.load_state_dict(weights['enc-apg-eta'])
    if lr is not None:
        assert isinstance(lr, float)
        optimizer =  torch.optim.Adam(list(enc_rws_eta.parameters())+list(enc_apg_z.parameters())+list(enc_apg_eta.parameters()),lr=lr, betas=(0.9, 0.99))
        return (enc_rws_eta, enc_apg_z, enc_apg_eta, generative), optimizer
    else: # testing
        for p in enc_rws_eta.parameters():
            p.requires_grad = False
        for p in enc_apg_z.parameters():
            p.requires_grad = False
        for p in enc_apg_eta.parameters():
            p.requires_grad = False
        return (enc_rws_eta, enc_apg_z, enc_apg_eta, generative)

def save_apg_models(models, save_version):
    """
    ==========
    saving function for APG samplers
    ==========
    """
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    checkpoint = {
        'enc-rws-eta' : enc_rws_eta.state_dict(),
        'enc-apg-z' : enc_apg_z.state_dict(),
        'enc-apg-eta' : enc_apg_eta.state_dict()
    }
    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    torch.save(checkpoint, "weights/cp-%s" % save_version)
    
def init_rws_models(K, D, num_hidden_z, CUDA, device, load_version=None, lr=None):
    """
    ==========
    initialization function for RWS method
    ==========
    """
    enc_rws_eta = Enc_rws_eta(K, D)
    enc_rws_z = Enc_apg_z(K, D, num_hidden_z)
    generative = Generative(K, D, CUDA, device)
    if CUDA:
        with torch.cuda.device(device):
            enc_rws_eta.cuda()
            enc_rws_z.cuda()
    if load_version is not None:
        weights = torch.load("../weights/cp-%s" % load_version)
        enc_rws_eta.load_state_dict(weights['enc-rws-eta'])
        enc_rws_z.load_state_dict(weights['enc-apg-z'])
    if lr is not None:
        assert isinstance(lr, float)
        optimizer =  torch.optim.Adam(list(enc_rws_eta.parameters())+list(enc_rws_z.parameters()),lr=lr, betas=(0.9, 0.99))
        return (enc_rws_eta, enc_rws_z, generative), optimizer
    else: # testing
        for p in enc_rws_eta.parameters():
            p.requires_grad = False
        for p in enc_rws_z.parameters():
            p.requires_grad = False
        return (enc_rws_eta, enc_rws_z, generative)

def save_rws_models(models, save_version):
    """
    ==========
    saving function for RWS method
    ==========
    """
    (enc_rws_eta, enc_rws_z, generative) = models
    checkpoint = {
        'enc-rws-eta' : enc_rws_eta.state_dict(),
        'enc-rws-z' : enc_rws_z.state_dict(),
    }
    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    torch.save(checkpoint, "weights/cp-%s" % save_version)

    
if __name__ == '__main__':
    import torch
    import numpy as np
    import argparse
    from apgs.resampler import Resampler
    from apgs.gmm.objectives import apg_objective, rws_objective
    parser = argparse.ArgumentParser('GMM Clustering Task')
    parser.add_argument('--data_dir', default='../../data/gmm/')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--budget', default=100, type=int)
    parser.add_argument('--num_sweeps', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--resample_strategy', default='systematic', choices=['systematic', 'multinomial'])
    parser.add_argument('--block_strategy', default='decomposed', choices=['decomposed', 'joint'])
    parser.add_argument('--num_clusters', default=3, type=int)
    parser.add_argument('--data_dim', default=2, type=int)
    parser.add_argument('--num_hidden', default=32, type=int)
    args = parser.parse_args()
    sample_size = int(args.budget / args.num_sweeps)
    CUDA = torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.device)
    
    data = torch.from_numpy(np.load(args.data_dir + 'ob.npy')).float() 
    assignments = torch.from_numpy(np.load(args.data_dir + 'assignment.npy')).float()
    print('Start training for gmm clustering task..')
    if args.num_sweeps == 1: ## rws method
        model_version = 'rws-gmm-num_samples=%s' % (sample_size)
        print('version='+ model_version)
        models, optimizer = init_rws_models(args.num_clusters, args.data_dim, args.num_hidden, CUDA, device, load_version=None, lr=args.lr)
        train(rws_objective, optimizer, models, data, assignments, args.num_epochs, sample_size, args.batch_size, CUDA, device)
        
    elif args.num_sweeps > 1: ## apg sampler
        model_version = 'apg-gmm-block=%s-num_sweeps=%s-num_samples=%s' % (args.block_strategy, args.num_sweeps, sample_size)
        print('version=' + model_version)
        models, optimizer = init_apg_models(args.num_clusters, args.data_dim, args.num_hidden, CUDA, device, load_version=None, lr=args.lr)
        resampler = Resampler(args.resample_strategy, sample_size, CUDA, device)
        train(apg_objective, optimizer, models, data, assignments, args.num_epochs, sample_size, args.batch_size, CUDA, device, num_sweeps=args.num_sweeps, block=args.block_strategy, resampler=resampler)
        
    else:
        raise ValueError
        