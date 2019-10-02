import probtorch
import torch
from os_coor import OS_coor
from enc_coor import Enc_coor
from enc_digit import Enc_digit
from dec_digit import Dec_digit

def Init_models(K, D, FRAME_PIXELS, DIGIT_PIXELS, HIDDEN_LIST, CUDA, device, lr, RESTORE=False, PATH=None):
    (NUM_HIDDEN_DIGIT, NUM_HIDDEN_COOR, Z_WHAT_DIM) = HIDDEN_LIST
    os_coor =OS_coor(K=K, D=D, num_pixels=(FRAME_PIXELS-DIGIT_PIXELS+1)**2, num_hidden=NUM_HIDDEN_COOR, CUDA=CUDA, device=device)
    enc_coor = Enc_coor(D=D, num_pixels=(FRAME_PIXELS-DIGIT_PIXELS+1)**2, num_hidden=NUM_HIDDEN_COOR, CUDA=CUDA, device=device)
    enc_digit = Enc_digit(num_pixels=DIGIT_PIXELS*DIGIT_PIXELS, num_hidden=NUM_HIDDEN_DIGIT, z_what_dim=Z_WHAT_DIM, CUDA=CUDA, device=device)
    dec_digit = Dec_digit(num_pixels=DIGIT_PIXELS**2, num_hidden=NUM_HIDDEN_DIGIT, z_what_dim=Z_WHAT_DIM)
    if CUDA:
        with torch.cuda.device(device):
            os_coor.cuda()
            enc_coor.cuda()
            enc_digit.cuda()
            dec_digit.cuda()
    if RESTORE:
        os_coor.load_state_dict(torch.load('../weights/os-coor-' + PATH))
        enc_coor.load_state_dict(torch.load('../weights/enc-coor-' + PATH))
        enc_digit.load_state_dict(torch.load('../weights/enc-digit-' + PATH))
        dec_digit.load_state_dict(torch.load('../weights/dec-digit-' + PATH))

    optimizer =  torch.optim.Adam(list(os_coor.parameters())+list(enc_coor.parameters())+list(enc_digit.parameters())+list(dec_digit.parameters()),lr=lr, betas=(0.9, 0.99))

    return (os_coor, enc_coor, enc_digit, dec_digit), optimizer


def Save_models(models, PATH):
    (os_coor, enc_coor, enc_digit, dec_digit) = models
    torch.save(os_coor.state_dict(), '../weights/os-coor-' + PATH)
    torch.save(enc_coor.state_dict(), '../weights/enc-coor-' + PATH)
    torch.save(enc_digit.state_dict(), '../weights/enc-digit-' + PATH)
    torch.save(dec_digit.state_dict(), '../weights/dec-digit-' + PATH)
