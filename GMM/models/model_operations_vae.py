from local_enc import *
from global_oneshot import *
from global_enc import *
import torch

def Init_models(K, D, num_hidden_local, CUDA, device, lr, RESTORE=False, PATH=None):
    # initialization
    os_eta = Oneshot_eta(K, D, CUDA, device)
    f_z = Enc_z(K, D, num_hidden_local, CUDA, device)
    if CUDA:
        with torch.cuda.device(device):
            os_eta.cuda()
            f_z.cuda()

    if RESTORE:
        os_eta.load_state_dict(torch.load("../weights/os-eta-%s" % PATH))
        f_z.load_state_dict(torch.load("../weights/f-z-%s" % PATH))
    optimizer =  torch.optim.Adam(list(os_eta.parameters())+list(f_z.parameters()),lr=lr, betas=(0.9, 0.99))
    return (os_eta, f_z), optimizer

def Save_models(models, path):
    (os_eta, f_z) = models
    torch.save(os_eta.state_dict(), "../weights/os-eta-%s" % path)
    torch.save(f_z.state_dict(), "../weights/f-z-%s" % path)
