import torch
from local_enc_state_v2 import *
from local_enc_angle import *
from global_lstm_no_ss import *
from decoder_semi import *

def Init_models_lstm(K, D, B, S, num_layers, hidden_list, RECON_SIGMA, CUDA, device, lr, RESTORE=False, PATH=None):
    (num_hidden_global, num_hidden_state, num_hidden_angle, num_hidden_dec, num_nss) = hidden_list
    # initialization
    os_mu = LSTM_eta(K, D, B, S, num_hidden_global, num_layers, CUDA, device)
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


def Save_models_lstm(models, PATH):
    (os_mu, f_state, f_angle, dec_x) = models
    torch.save(os_mu.state_dict(), '../weights/os-mu-' + PATH)
    torch.save(f_state.state_dict(), '../weights/f-state-' + PATH)
    torch.save(f_angle.state_dict(), '../weights/f-angle-' + PATH)
    torch.save(dec_x.state_dict(), '../weights/dec-x-' + PATH)

    
def Init_models_test(K, D, B, S, num_layers, hidden_list, RECON_SIGMA, CUDA, device, lr, RESTORE=False, PATH=None):
    (num_hidden_global, num_hidden_state, num_hidden_angle, num_hidden_dec, num_nss) = hidden_list
    # initialization
    os_mu = LSTM_eta(K, D, B, S, num_hidden_global, num_layers, CUDA, device)
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
    for p in os_mu.parameters():
        p.requires_grad = False
        
    for p in f_state.parameters():
        p.requires_grad = False
        
    for p in f_angle.parameters():
        p.requires_grad = False
        
    for p in dec_x.parameters():
        p.requires_grad = False
    return (os_mu, f_state, f_angle, dec_x)