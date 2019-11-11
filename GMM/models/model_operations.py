import torch
from enc_rws_eta import Enc_rws_eta
from enc_apg_eta import Enc_apg_eta
from enc_apg_z import Enc_apg_z
from generative import Generative

def init_models(model_params, CUDA, DEVICE, LOAD_VERSION=None, LR=None):
    (K, D, num_hidden_local) = model_params
    enc_rws_eta = Enc_rws_eta(K, D)
    enc_apg_z = Enc_apg_z(K, D, num_hidden_local)
    enc_apg_eta = Enc_apg_eta(K, D)
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
        return (enc_rws_eta, enc_apg_z, enc_apg_eta, generative)

def save_models(models, SAVE_VERSION):
    (enc_rws_eta, enc_apg_z, enc_apg_eta, generative) = models
    torch.save(enc_rws_eta.state_dict(), "../weights/enc-rws-eta-%s" % SAVE_VERSION)
    torch.save(enc_apg_z.state_dict(), "../weights/enc-apg-z-%s" % SAVE_VERSION)
    torch.save(enc_apg_eta.state_dict(), "../weights/enc-apg-eta-%s" % SAVE_VERSION)
