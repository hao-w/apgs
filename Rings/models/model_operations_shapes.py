import torch
from global_oneshot_mu import *
<<<<<<< HEAD
from local_oneshot_state_shapes import *
from local_enc_angle import *
from global_enc_mu_shapes import *
=======
from local_oneshot_state import *
from local_enc_angle import *
from global_enc_mu import *
>>>>>>> 275993c21a43e1a2593cc2771950a4fa33b00d4c
from decoder_shapes import *

def Init_models(K, D, num_hidden, num_hidden_local, num_nss, RECON_SIGMA, CUDA, device, lr, RESTORE=False, PATH=None):
    # initialization
    os_mu = Oneshot_mu(K, D, num_hidden, num_nss, CUDA, device)
    f_state = Oneshot_state(K, D, num_hidden_local, CUDA, device)
    f_angle = Enc_angle(D, num_hidden, CUDA, device)
    f_mu = Enc_mu(K, D, num_hidden, num_nss, CUDA, device)
    dec_x = Dec_x(D, num_hidden, RECON_SIGMA, CUDA, device)

    b_state = Oneshot_state(K, D, num_hidden_local, CUDA, device)
    b_angle = Enc_angle(D, num_hidden, CUDA, device)
    b_mu = Enc_mu(K, D, num_hidden, num_nss, CUDA, device)
    if CUDA:
        with torch.cuda.device(device):
            os_mu.cuda()
            f_state.cuda()
            f_angle.cuda()
            f_mu.cuda()
            dec_x.cuda()
            b_state.cuda()
            b_angle.cuda()
            b_mu.cuda()
    if RESTORE:
        os_mu.load_state_dict(torch.load('../weights/os-mu-' + PATH))
        f_state.load_state_dict(torch.load('../weights/f-state-' + PATH))
        f_angle.load_state_dict(torch.load('../weights/f-angle-' + PATH))
        f_mu.load_state_dict(torch.load('../weights/f-mu-' + PATH))
        dec_x.load_state_dict(torch.load('../weights/dec-x-' + PATH))

        b_state.load_state_dict(torch.load('../weights/b-state-' + PATH))
        b_angle.load_state_dict(torch.load('../weights/b-angle-' + PATH))
        b_mu.load_state_dict(torch.load('../weights/b-mu-' + PATH))
    optimizer =  torch.optim.Adam(list(os_mu.parameters())+list(f_state.parameters())+list(f_angle.parameters())+list(f_mu.parameters())+list(b_state.parameters())+list(b_angle.parameters())+list(b_mu.parameters())+list(dec_x.parameters()),lr=lr, betas=(0.9, 0.99))

    # optimizer =  torch.optim.Adam(list(os_mu.parameters())+list(f_state.parameters())+list(f_angle.parameters())+list(f_mu.parameters())+list(dec_x.parameters()),lr=lr, betas=(0.9, 0.99))
    return (os_mu, f_state, f_angle, f_mu, b_state, b_angle, b_mu, dec_x), optimizer

def Save_models(models, path):
    (os_mu, f_state, f_angle, f_mu, b_state, b_angle, b_mu, dec_x) = models
    torch.save(os_mu.state_dict(), '../weights/os-mu-' + path)
    torch.save(f_state.state_dict(), '../weights/f-state-' + path)
    torch.save(f_angle.state_dict(), '../weights/f-angle-' + path)
    torch.save(f_mu.state_dict(), '../weights/f-mu-' + path)
    torch.save(dec_x.state_dict(), '../weights/dec-x-' + path)

    torch.save(b_state.state_dict(), '../weights/b-state-' + path)
    torch.save(b_angle.state_dict(), '../weights/b-angle-' + path)
    torch.save(b_mu.state_dict(), '../weights/b-mu-' + path)
