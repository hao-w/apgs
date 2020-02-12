import torch
from enc_rws_eta_neural import Enc_rws_eta
from enc_apg_eta_neural import Enc_apg_eta
from enc_apg_z import Enc_apg_z
from generative import Generative

def init_model(model_params, CUDA, DEVICE, LOAD_VERSION=None, LR=None):
    """
    ==========
    initialization function for APG samplers
    ==========
    """
    (K, D, num_hidden_eta, num_hidden_z) = model_params
    enc_rws_eta = Enc_rws_eta(K, D, num_hidden_eta)
    enc_apg_z = Enc_apg_z(K, D, num_hidden_z)
    enc_apg_eta = Enc_apg_eta(K, D, num_hidden_eta)
    generative = Generative(K, D, CUDA, DEVICE)
    if CUDA:
        with torch.cuda.device(DEVICE):
            enc_rws_eta.cuda()
            enc_apg_z.cuda()
            enc_apg_eta.cuda()
    if LOAD_VERSION is not None:
        enc_rws_eta.load_state_dict(torch.load("../weights/enc-rws-eta-%s" % LOAD_VERSION))
        enc_apg_z.load_state_dict(torch.load("../weights/enc-apg-z-%s" % LOAD_VERSION))
        enc_apg_eta.load_state_dict(torch.load("../weights/enc-apg-eta-%s" % LOAD_VERSION))
    if LR is not None: # training
        assert isinstance(LR, float)
        optimizer =  torch.optim.Adam(list(enc_rws_eta.parameters())+list(enc_apg_z.parameters())+list(enc_apg_eta.parameters()),lr=LR, betas=(0.9, 0.99))
        return (enc_rws_eta, enc_apg_z, enc_apg_eta, generative), optimizer
    else: # testing
        for p in enc_rws_eta.parameters():
            p.requires_grad = False
        for p in enc_apg_z.parameters():
            p.requires_grad = False
        for p in enc_apg_eta.parameters():
            p.requires_grad = False
        return (enc_rws_eta, enc_apg_z, enc_apg_eta, generative)

def save_model(model, SAVE_VERSION):
    """
    ==========
    saving function for APG samplers
    ==========
    """
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = model
    torch.save(enc_rws_eta.state_dict(), "../weights/enc-rws-eta-%s" % SAVE_VERSION)
    torch.save(enc_apg_z.state_dict(), "../weights/enc-apg-z-%s" % SAVE_VERSION)
    torch.save(enc_apg_eta.state_dict(), "../weights/enc-apg-eta-%s" % SAVE_VERSION)
