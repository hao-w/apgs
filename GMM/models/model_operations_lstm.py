from global_lstm_no_ss import *
from local_enc import *
import torch

def Init_models(K, D, B, S, NUM_HIDDEN_GLOBAL, NUM_HIDDEN_LOCAL, NUM_LAYERS, CUDA, device, lr, RESTORE=False, PATH=None):
    # initialization
    os_eta = LSTM_eta(K, D, B, S, num_hidden_global=NUM_HIDDEN_GLOBAL, num_layers=NUM_LAYERS, CUDA=CUDA, device=device)
    f_z = Enc_z(K, D, num_hidden_local=NUM_HIDDEN_LOCAL, CUDA=CUDA, device=device)
    if CUDA:
        with torch.cuda.device(device):
            os_eta.cuda()
            f_z.cuda()
    optimizer =  torch.optim.Adam(list(f_z.parameters()) + list(os_eta.parameters()),lr=LEARNING_RATE, betas=(0.9, 0.99))
    if RESTORE:
        os_eta.load_state_dict(torch.load("../weights/os-eta-%s" % PATH))
        f_z.load_state_dict(torch.load("../weights/f-z-%s" % PATH))
    optimizer =  torch.optim.Adam(list(os_eta.parameters())+list(f_z.parameters()),lr=lr, betas=(0.9, 0.99))
    return (os_eta, f_z), optimizer

def Save_models(models, path):
    (os_eta, f_z) = models
    torch.save(os_eta.state_dict(), "../weights/os-eta-%s" % path)
    torch.save(f_z.state_dict(), "../weights/f-z-%s" % path)
