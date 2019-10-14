from local_enc import *
from global_oneshot import *
from global_enc import *
import torch

def Init_models(K, D, num_hidden_local, CUDA, device, lr, RESTORE=False, PATH=None):
    # initialization
    os_eta = Oneshot_eta(K, D, CUDA, device)

    f_z = Enc_z(K, D, num_hidden_local, CUDA, device)
    f_eta = Enc_eta(K, D, CUDA, device)

    # b_z = Enc_z(K, D, num_hidden_local, CUDA, device)
    # b_eta = Enc_eta(K, D, CUDA, device)

    if CUDA:
        with torch.cuda.device(device):
            os_eta.cuda()
            f_z.cuda()
            f_eta.cuda()
            # b_z.cuda()
            # b_eta.cuda()
    if RESTORE:
        os_eta.load_state_dict(torch.load("../weights/os-eta-%s" % PATH))
        f_z.load_state_dict(torch.load("../weights/f-z-%s" % PATH))
        f_eta.load_state_dict(torch.load("../weights/f-eta-%s" % PATH))
        # b_z.load_state_dict(torch.load("../weights/b-z-%s" % PATH))
        # b_eta.load_state_dict(torch.load("../weights/b-eta-%s" % PATH))
    optimizer =  torch.optim.Adam(list(os_eta.parameters())+list(f_z.parameters())+list(f_eta.parameters()),lr=lr, betas=(0.9, 0.99))
    # optimizer =  torch.optim.Adam(list(os_eta.parameters())+list(f_z.parameters())+list(f_eta.parameters())+list(b_z.parameters())+list(b_eta.parameters()),lr=lr, betas=(0.9, 0.99))
    return (os_eta, f_z, f_eta), optimizer

def Save_models(models, path):
    (os_eta, f_z, f_eta) = models
    torch.save(os_eta.state_dict(), "../weights/os-eta-%s" % path)
    torch.save(f_z.state_dict(), "../weights/f-z-%s" % path)
    torch.save(f_eta.state_dict(), "../weights/f-eta-%s" % path)

    # torch.save(b_z.state_dict(), "../weights/b-z-%s" % path)
    # torch.save(b_eta.state_dict(), "../weights/b-eta-%s" % path)

def Init_models_test(K, D, num_hidden_local, CUDA, device, lr, RESTORE=False, PATH=None):
    # initialization
    os_eta = Oneshot_eta(K, D, CUDA, device)

    f_z = Enc_z(K, D, num_hidden_local, CUDA, device)
    f_eta = Enc_eta(K, D, CUDA, device)

    if CUDA:
        with torch.cuda.device(device):
            os_eta.cuda()
            f_z.cuda()
            f_eta.cuda()

    if RESTORE:
        os_eta.load_state_dict(torch.load("../weights/os-eta-%s" % PATH))
        f_z.load_state_dict(torch.load("../weights/f-z-%s" % PATH))
        f_eta.load_state_dict(torch.load("../weights/f-eta-%s" % PATH))
    
    for p in os_eta.parameters():
        p.requires_grad =False
    for p in f_eta.parameters():
        p.requires_grad =False
    for p in f_z.parameters():
        p.requires_grad =False
    return (os_eta, f_z, f_eta)