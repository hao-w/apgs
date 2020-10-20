import torch
from BMNIST.modules.enc_coor import Enc_coor
from BMNIST.modules.enc_digit import Enc_digit
from BMNIST.modules.dec_digit import Dec_digit
from BMNIST.modules.dec_coor import Dec_coor

def init_model(frame_pixels, digit_pixels, num_hidden_digit, num_hidden_coor, z_where_dim, z_what_dim, CUDA, DEVICE, LOAD_VERSION, LR):
    enc_coor = Enc_coor(num_pixels=(frame_pixels-digit_pixels+1)**2, num_hidden=num_hidden_coor, z_where_dim=z_where_dim)
    dec_coor = Dec_coor(z_where_dim=z_where_dim, CUDA=CUDA, DEVICE=DEVICE)
    enc_digit = Enc_digit(num_pixels=digit_pixels**2, num_hidden=num_hidden_digit, z_what_dim=z_what_dim)
    dec_digit = Dec_digit(num_pixels=digit_pixels**2, num_hidden=num_hidden_digit, z_what_dim=z_what_dim, CUDA=CUDA, DEVICE=DEVICE)
    if CUDA:
        with torch.cuda.device(DEVICE):
            enc_coor.cuda()
            enc_digit.cuda()
            dec_digit.cuda()
    if LOAD_VERSION is not None: ## if initialization is at training time
        enc_coor.load_state_dict(torch.load('../weights/enc-coor-%s' % LOAD_VERSION))
        enc_digit.load_state_dict(torch.load('../weights/enc-digit-%s' % LOAD_VERSION))
        dec_digit.load_state_dict(torch.load('../weights/dec-digit-%s' % LOAD_VERSION))
    if LR is not None:
        optimizer =  torch.optim.Adam(list(enc_coor.parameters())+
                                        # list(dec_coor.parameters())+
                                        list(enc_digit.parameters())+
                                        list(dec_digit.parameters()),
                                        lr=LR,
                                        betas=(0.9, 0.99))

        return (enc_coor, dec_coor, enc_digit, dec_digit), optimizer
    else: ## if initialization is at testing time
        for p in enc_coor.parameters():
            p.requires_grad = False
        for p in enc_digit.parameters():
            p.requires_grad = False
        for p in dec_digit.parameters():
            p.requires_grad = False
    return (enc_coor, dec_coor, enc_digit, dec_digit)


def save_model(model, SAVE_VERSION):
    (enc_coor, dec_coor, enc_digit, dec_digit) = model
    torch.save(enc_coor.state_dict(), '../weights/enc-coor-%s' % SAVE_VERSION)
    # torch.save(dec_coor.state_dict(), '../weights/dec-coor-%s' + SAVE_VERSION)
    torch.save(enc_digit.state_dict(), '../weights/enc-digit-%s' % SAVE_VERSION)
    torch.save(dec_digit.state_dict(), '../weights/dec-digit-%s' % SAVE_VERSION)
