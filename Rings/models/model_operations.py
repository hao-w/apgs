import torch
from global_oneshot_mu import *
from local_enc_state_v2 import *
from local_enc_angle import *
from global_enc_mu_v2 import *
from decoder_semi import *

def Init_models(K, D, hidden_list, RECON_SIGMA, CUDA, device, lr, RESTORE=False, PATH=None, NAME='APG'):
    (num_hidden_global, num_hidden_state, num_hidden_angle, num_hidden_dec, num_nss) = hidden_list
    # initialization
    if NAME == 'APG':
        os_mu = Oneshot_mu(K, D, num_hidden_global, num_nss, CUDA, device)
        f_state = Enc_state(K, D, num_hidden_state, CUDA, device)
        f_angle = Enc_angle(D, num_hidden_angle, CUDA, device)
        f_mu = Enc_mu(K, D, num_hidden_global, num_nss, CUDA, device)
        dec_x = Dec_x(K, D, num_hidden_dec, RECON_SIGMA, CUDA, device)
        if CUDA:
            with torch.cuda.device(device):
                os_mu.cuda()
                f_state.cuda()
                f_angle.cuda()
                f_mu.cuda()
                dec_x.cuda()
        if RESTORE:
            os_mu.load_state_dict(torch.load('../weights/os-mu-' + PATH))
            f_state.load_state_dict(torch.load('../weights/f-state-' + PATH))
            f_angle.load_state_dict(torch.load('../weights/f-angle-' + PATH))
            f_mu.load_state_dict(torch.load('../weights/f-mu-' + PATH))
            dec_x.load_state_dict(torch.load('../weights/dec-x-' + PATH))
        optimizer =  torch.optim.Adam(list(os_mu.parameters())+list(f_state.parameters())+list(f_angle.parameters())+list(f_mu.parameters())+list(dec_x.parameters()),lr=lr, betas=(0.9, 0.99))
        return (os_mu, f_state, f_angle, f_mu, dec_x), optimizer
    elif NAME == 'VAE':
        os_mu = Oneshot_mu(K, D, num_hidden_global, num_nss, CUDA, device)
        f_state = Enc_state(K, D, num_hidden_state, CUDA, device)
        f_angle = Enc_angle(D, num_hidden_angle, CUDA, device)
        dec_x = Dec_x(K, D, num_hidden_dec, RECON_SIGMA, CUDA, device)
        if CUDA:
            with torch.cuda.device(device):
                os_mu.cuda()
                f_state.cuda()
                f_angle.cuda()
                dec_x.cuda()
        if RESTORE:
            os_mu.load_state_dict(torch.load('../weights/os-mu-' + PATH))
            f_state.load_state_dict(torch.load('../weights/f-state-' + PATH))
            f_angle.load_state_dict(torch.load('../weights/f-angle-' + PATH))
            dec_x.load_state_dict(torch.load('../weights/dec-x-' + PATH))
        optimizer =  torch.optim.Adam(list(os_mu.parameters())+list(f_state.parameters())+list(f_angle.parameters())+list(dec_x.parameters()),lr=lr, betas=(0.9, 0.99))
        return (os_mu, f_state, f_angle, dec_x), optimizer
    else:
        print('ERROR : Undefined model name!')

def Save_models(models, path, NAME='APG'):
    if NAME == 'APG':
        (os_mu, f_state, f_angle, f_mu, dec_x) = models
        torch.save(os_mu.state_dict(), '../weights/os-mu-' + path)
        torch.save(f_state.state_dict(), '../weights/f-state-' + path)
        torch.save(f_angle.state_dict(), '../weights/f-angle-' + path)
        torch.save(f_mu.state_dict(), '../weights/f-mu-' + path)
        torch.save(dec_x.state_dict(), '../weights/dec-x-' + path)
    elif NAME == 'VAE':
        (os_mu, f_state, f_angle, dec_x) = models
        torch.save(os_mu.state_dict(), '../weights/os-mu-' + path)
        torch.save(f_state.state_dict(), '../weights/f-state-' + path)
        torch.save(f_angle.state_dict(), '../weights/f-angle-' + path)
        torch.save(dec_x.state_dict(), '../weights/dec-x-' + path)
    else:
        print('ERROR : Undefined model name!')
