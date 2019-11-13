import torch
from enc_rws_mu import Enc_rws_mu
from enc_apg_local import Enc_apg_local
from enc_apg_mu import Enc_apg_mu
from decoder import Dec

def init_model(model_params, CUDA, DEVICE, LOAD_VERSION=None, LR=None):
    """
    ==========
    initialization funtion of APG samplers
    ==========
    """
    (K, D, num_hidden_global, num_nss, num_hidden_state, num_hidden_angle, num_hidden_dec, recon_sigma) = model_params
    enc_rws_mu = Enc_rws_mu(K, D, num_hidden_global, num_nss)
    enc_apg_local = Enc_apg_local(K, D, num_hidden_state, num_hidden_angle)
    enc_apg_mu = Enc_apg_mu(K, D, num_hidden_global, num_nss)
    dec = Dec(K, D, num_hidden_dec, recon_sigma, CUDA, DEVICE)
    if CUDA:
        with torch.cuda.device(DEVICE):
            enc_rws_mu.cuda()
            enc_apg_local.cuda()
            enc_apg_mu.cuda()
            dec.cuda()
    if LOAD_VERSION is not None:
        enc_rws_mu.load_state_dict(torch.load('../weights/enc-rws-mu-%s' % LOAD_VERSION))
        enc_apg_local.load_state_dict(torch.load('../weights/enc-apg-local-%s' % LOAD_VERSION))
        enc_apg_mu.load_state_dict(torch.load('../weights/enc-apg-mu-%s' % LOAD_VERSION))
        dec.load_state_dict(torch.load('../weights/dec-%s' % LOAD_VERSION))
    if LR is not None:
        assert isinstance(LR, float)
        optimizer =  torch.optim.Adam(list(enc_rws_mu.parameters())+list(enc_apg_local.parameters())+list(enc_apg_mu.parameters())+list(dec.parameters()),lr=LR, betas=(0.9, 0.99))
        return (enc_rws_mu, enc_apg_local, enc_apg_mu, dec), optimizer
    else:
        for p in enc_rws_mu.parameters():
            p.requires_grad = False
        for p in enc_apg_local.parameters():
            p.requires_grad = False
        for p in enc_apg_mu.parameters():
            p.requires_grad = False
        for p in dec.parameters():
            p.requires_grad = False
        return (enc_rws_mu, enc_apg_local, enc_apg_mu, dec)

def save_model(model, SAVE_VERSION):
    """
    ==========
    saving function for APG samplers
    ==========
    """
    (enc_rws_mu, enc_apg_local, enc_apg_mu, dec) = model
    torch.save(enc_rws_mu.state_dict(), '../weights/enc-rws-mu-%s' % SAVE_VERSION)
    torch.save(enc_apg_local.state_dict(), '../weights/enc-apg-local-%s' % SAVE_VERSION)
    torch.save(enc_apg_mu.state_dict(), '../weights/enc-apg-mu-%s' % SAVE_VERSION)
    torch.save(dec.state_dict(), '../weights/dec-%s' % SAVE_VERSION)
