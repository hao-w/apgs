import torch
from global_oneshot_mu import *
from local_oneshot_state import *
from local_enc_angle import *
from decoder_semi import *

def Init_models(K, D, num_hidden, num_hidden_local, num_nss, RECON_SIGMA, CUDA, device, lr, RESTORE=False, PATH=None):
    # initialization
    os_mu = Oneshot_mu(K, D, num_hidden, num_nss, CUDA, device)
    f_state = Oneshot_state(K, D, num_hidden_local, CUDA, device)
    f_angle = Enc_angle(D, num_hidden, CUDA, device)
    dec_x = Dec_x(D, num_hidden, RECON_SIGMA, CUDA, device)

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

def Save_models(models, path):
    (os_mu, f_state, f_angle, dec_x) = models
    torch.save(os_mu.state_dict(), '../weights/os-mu-' + path)
    torch.save(f_state.state_dict(), '../weights/f-state-' + path)
    torch.save(f_angle.state_dict(), '../weights/f-angle-' + path)
    torch.save(dec_x.state_dict(), '../weights/dec-x-' + path)
