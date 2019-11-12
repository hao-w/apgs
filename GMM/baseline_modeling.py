import torch
from enc_rws_eta import Enc_rws_eta # mlp architecture
from enc_rws_eta_lstm import Enc_rws_eta_lstm # lstm architecture
from enc_apg_z import Enc_apg_z
from generative import Generative

def init_model(architecture, model_params, CUDA, DEVICE, LOAD_VERSION=None, LR=None):
    """
    ==========
    initialization function for baselines
    ==========
    """
    if architecture == 'mlp':
        (K, D, num_hidden_z) = model_params
        enc_rws_eta = Enc_rws_eta(K, D)
    elif architecture == 'lstm':
        (K, D, num_hidden_z, num_hidden_lstm, B, S, num_layers) = model_params
        enc_rws_eta = Enc_rws_eta_lstm(K, D, num_hidden_lstm, B, S, num_layers, CUDA, DEVICE)
    else:
        print('ERROR! unexpected architecture name.')
    enc_rws_z = Enc_apg_z(K, D, num_hidden_z)
    generative = Generative(K, D, CUDA, DEVICE)
    if CUDA:
        with torch.cuda.device(DEVICE):
            enc_rws_eta.cuda()
            enc_rws_z.cuda()
    if LOAD_VERSION is not None:
        enc_rws_eta.load_state_dict(torch.load("../weights/enc-rws-eta-%s" % LOAD_VERSION))
        enc_rws_z.load_state_dict(torch.load("../weights/enc-rws-z-%s" % LOAD_VERSION))
    if LR is not None: # training
        assert isinstance(LR, float)
        optimizer =  torch.optim.Adam(list(enc_rws_eta.parameters())+list(enc_rws_z.parameters()),lr=LR, betas=(0.9, 0.99))
        return (enc_rws_eta, enc_rws_z, generative), optimizer
    else: # testing
        for p in enc_rws_eta.parameters():
            p.requires_grad = False
        for p in enc_rws_z.parameters():
            p.requires_grad = False
        return (enc_rws_eta, enc_rws_z, generative)

def save_model(model, SAVE_VERSION):
    """
    ==========
    saving function for baselines
    ==========
    """
    (enc_rws_eta, enc_rws_z, generative) = model
    torch.save(enc_rws_eta.state_dict(), "../weights/enc-rws-eta-%s" % SAVE_VERSION)
    torch.save(enc_rws_z.state_dict(), "../weights/enc-rws-z-%s" % SAVE_VERSION)
