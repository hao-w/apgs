import os
import torch
import time
import numpy as np
from random import shuffle
from apgs.bshape.models import Enc_coor, Dec_coor, Enc_digit, Dec_digit
from apgs.bshape.objectives import apg_objective

def train(optimizer, models, AT, resampler, num_sweeps, data_paths, shape_mean, K, num_epochs, sample_size, batch_size, CUDA, device, model_version):
    """
    training function of apg samplers
    """
    result_flags = {'loss_required' : True, 'ess_required' : True, 'mode_required' : False, 'density_required': True}
    shape_mean = shape_mean.repeat(sample_size, batch_size, K, 1, 1)
    for epoch in range(num_epochs):
        shuffle(data_paths)
        for group, data_path in enumerate(data_paths):
            time_start = time.time()
            metrics = dict()
            data = torch.from_numpy(np.load(data_path)).float()
            num_batches = int(data.shape[0] / batch_size)
            seq_indices = torch.randperm(data.shape[0])
            for b in range(num_batches):
                optimizer.zero_grad()
                frames = data[seq_indices[b*batch_size : (b+1)*batch_size]].repeat(sample_size, 1, 1, 1, 1)
                if CUDA:
                    with torch.cuda.device(device):
                        frames = frames.cuda()
                        shape_mean = shape_mean.cuda()
                trace = apg_objective(models, AT, frames, K, result_flags, num_sweeps, resampler, shape_mean)
                loss_phi = trace['loss_phi'].sum()
                loss_theta = trace['loss_theta'].sum()
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
                if 'ess' in metrics:
                    metrics['ess'] += trace['ess'][-1].mean().item()
                else:
                    metrics['ess'] = trace['ess'][-1].mean().item()
                if 'density' in metrics:
                    metrics['density'] += trace['density'][-1].mean().item()
                else:
                    metrics['density'] = trace['density'][-1].mean().item()
            save_models(models, model_version)
            metrics_print = ",  ".join(['%s: %.4f' % (k, v/num_batches) for k, v in metrics.items()])
            if not os.path.exists('results/'):
                os.makedirs('results/')
            if epoch == 0 and group == 0:
                log_file = open('results/log-' + model_version + '.txt', 'w+')
            else:
                log_file = open('results/log-' + model_version + '.txt', 'a+')
            time_end = time.time()
            print("(%ds) Epoch=%d, Group=%d, " % (time_end - time_start, epoch, group) + metrics_print, file=log_file)
            log_file.close()
            print("Epoch=%d, Group=%d completed in (%ds),  " % (epoch, group, time_end - time_start))
            
def init_models(AT, frame_pixels, digit_pixels, num_hidden_digit, num_hidden_coor, z_where_dim, z_what_dim, CUDA, device, load_version, lr):
    enc_coor = Enc_coor(num_pixels=(frame_pixels-digit_pixels+1)**2, num_hidden=num_hidden_coor, z_where_dim=z_where_dim, AT=AT)
    dec_coor = Dec_coor(z_where_dim=z_where_dim, CUDA=CUDA, device=device)
    enc_digit = Enc_digit(num_pixels=digit_pixels**2, num_hidden=num_hidden_digit, z_what_dim=z_what_dim, AT=AT)
    dec_digit = Dec_digit(num_pixels=digit_pixels**2, num_hidden=num_hidden_digit, z_what_dim=z_what_dim, AT=AT, CUDA=CUDA, device=device)
    if CUDA:
        with torch.cuda.device(device):
            enc_coor.cuda()
            enc_digit.cuda()
            dec_digit.cuda()
            
    if load_version is not None: 
        weights = torch.load("weights/cp-%s" % load_version)
        enc_coor.load_state_dict(weights['enc-coor'])
        enc_digit.load_state_dict(weights['enc-digit'])
        dec_digit.load_state_dict(weights['dec-digit'])
    if lr is not None:
        optimizer =  torch.optim.Adam(list(enc_coor.parameters())+
                                        list(enc_digit.parameters())+
                                        list(dec_digit.parameters()),
                                        lr=lr,
                                        betas=(0.9, 0.99))
#         optimizer =  torch.optim.SGD(list(enc_coor.parameters())+
#                                         list(enc_digit.parameters())+
#                                         list(dec_digit.parameters()),
#                                         lr=lr)
        return (enc_coor, dec_coor, enc_digit, dec_digit), optimizer
#     else: 
#         for p in enc_coor.parameters():
#             p.requires_grad = False
#         for p in enc_digit.parameters():
#             p.requires_grad = False
#         for p in dec_digit.parameters():
#             p.requires_grad = False
    return (enc_coor, dec_coor, enc_digit, dec_digit)

def save_models(models, save_version):
    (enc_coor, dec_coor, enc_digit, dec_digit) = models
    checkpoint = {
        'enc-coor' : enc_coor.state_dict(),
        'enc-digit' : enc_digit.state_dict(),
        'dec-digit' : dec_digit.state_dict()
    }
    if not os.path.exists('weights/'):
        os.makedirs('weights/')
    torch.save(checkpoint, 'weights/cp-%s' % save_version)
    
if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    import argparse
    from apgs.resampler import Resampler
    from apgs.bmnist.affine_transformer import Affine_Transformer
    from apgs.snr import EMA, set_seed
    parser = argparse.ArgumentParser('Bouncing Shapes')
    parser.add_argument('--data_dir', default='../../data/bshape/')
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--budget', default=100, type=int)
    parser.add_argument('--num_sweeps', default=2, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--resample_strategy', default='systematic', choices=['systematic', 'multinomial'])
    parser.add_argument('--num_objects', default=2, type=int)
    parser.add_argument('--timesteps', default=10, type=int)
    parser.add_argument('--frame_pixels', default=40, type=int)
    parser.add_argument('--shape_pixels', default=10, type=int)
    parser.add_argument('--num_hidden_digit', default=200, type=int)
    parser.add_argument('--num_hidden_coor', default=200, type=int)
    parser.add_argument('--z_where_dim', default=2, type=int)
    parser.add_argument('--z_what_dim', default=10, type=int)
    args = parser.parse_args()
    sample_size = int(args.budget / args.num_sweeps)
    set_seed(0) 
    CUDA = torch.cuda.is_available()
    device = torch.device('cuda:%d' % args.device)
    if args.num_sweeps == 1: ## rws method
        model_version = 'rws-bshape-num_objects=%s-num_samples=%s' % (args.num_objects, sample_size)
    elif args.num_sweeps > 1: ## apg sampler
        model_version = 'apg-bshape-num_objects=%s-num_sweeps=%s-num_samples=%s' % (args.num_objects, args.num_sweeps, sample_size)
    else:
        raise ValueError
      
    data_paths = []
    for file in os.listdir(args.data_dir + '%dobjects/train/' % args.num_objects):
        data_paths.append(os.path.join(args.data_dir, '%dobjects/train' % args.num_objects, file))
        
    shape_mean = torch.from_numpy(np.load('shape_mean.npy')).float()
    
    AT = Affine_Transformer(args.frame_pixels, args.shape_pixels, CUDA, device)
    resampler = Resampler(args.resample_strategy, sample_size, CUDA, device)
    
    models, optimizer = init_models(AT, args.frame_pixels, args.shape_pixels, args.num_hidden_digit, args.num_hidden_coor, args.z_where_dim, args.z_what_dim, CUDA, device, load_version=None, lr=args.lr)
    print('Start training for bshape tracking task..')
    print('version=' + model_version)  
    train(optimizer, models, AT, resampler, args.num_sweeps, data_paths, shape_mean, args.num_objects, args.num_epochs, sample_size, args.batch_size, CUDA, device, model_version)        
