import torch
from global_oneshot_mu import *
from local_oneshot_state import *
from local_enc_angle import *
from global_enc_mu import *
from decoder_semi import *

def init_models(K, D, num_hidden, num_hidden_local, num_nss, RECON_SIGMA, CUDA, device, RESTORE=False, PATH=None):
    # initialization
    oneshot_mu = Oneshot_mu(K, D, num_hidden, num_nss, CUDA, device)
    oneshot_state = Oneshot_state(K, D, num_hidden_local, CUDA, device)
    enc_angle = Enc_angle(D, num_hidden, CUDA, device)
    enc_mu = Enc_mu(K, D, num_hidden, num_nss, CUDA, device)
    dec_x = Dec_x(D, num_hidden, RECON_SIGMA, CUDA, device)
    if CUDA:
        with torch.cuda.device(device):
            oneshot_mu.cuda()
            oneshot_state.cuda()
            enc_angle.cuda()
            enc_mu.cuda()
            dec_x.cuda()
    if RESTORE:
        oneshot_mu.load_state_dict(torch.load('../results/oneshot-mu-' + PATH))
        oneshot_state.load_state_dict(torch.load('../results/oneshot-state-' + PATH))
        enc_angle.load_state_dict(torch.load('../results/enc-angle-' + PATH))
        enc_mu.load_state_dict(torch.load('../results/enc-mu-' + PATH))
        dec_x.load_state_dict(torch.load('../results/dec-x-' + PATH))
    return (oneshot_mu, oneshot_state, enc_angle, enc_mu, dec_x)
