import probtorch
import torch
from affine_transformer import Affine_Transformer
from enc_coor import Enc_coor
from enc_digit import Enc_digit
from dec_digit import Dec_digit
from dec_coor import Dec_coor

def Init_models(K, D, FRAME_PIXELS, DIGIT_PIXELS, HIDDEN_LIST, Sigma0, CUDA, DEVICE, lr, RESTORE=False, PATH=None):
    (NUM_HIDDEN_DIGIT, NUM_HIDDEN_COOR, Z_WHAT_DIM) = HIDDEN_LIST
    AT = Affine_Transformer(digit_size=28, frame_size=64, CUDA=CUDA, DEVICE=DEVICE)
    enc_coor = Enc_coor(D=D, num_pixels=(FRAME_PIXELS-DIGIT_PIXELS+1)**2, num_hidden=NUM_HIDDEN_COOR)
    enc_digit = Enc_digit(num_pixels=DIGIT_PIXELS*DIGIT_PIXELS, num_hidden=NUM_HIDDEN_DIGIT, z_what_dim=Z_WHAT_DIM, CUDA=CUDA, DEVICE=DEVICE)
    dec_digit = Dec_digit(num_pixels=DIGIT_PIXELS**2, num_hidden=NUM_HIDDEN_DIGIT, z_what_dim=Z_WHAT_DIM, AT=AT)
    dec_coor = Dec_coor(D=D, Sigma0=Sigma0, CUDA=CUDA, DEVICE=DEVICE)
    if CUDA:
        with torch.cuda.device(DEVICE):
            enc_coor.cuda()
            enc_digit.cuda()
            dec_digit.cuda()
            # dec_coor.cuda()
    if RESTORE:
        enc_coor.load_state_dict(torch.load('../weights/enc-coor-' + PATH))
        enc_digit.load_state_dict(torch.load('../weights/enc-digit-' + PATH))
        dec_digit.load_state_dict(torch.load('../weights/dec-digit-' + PATH))
        # dec_coor.load_state_dict(torch.load('../weights/dec-coor-' + PATH))

    optimizer =  torch.optim.Adam(list(enc_coor.parameters())+
                                    # list(dec_coor.parameters())+
                                    list(enc_digit.parameters())+
                                    list(dec_digit.parameters()),
                                    lr=lr,
                                    betas=(0.9, 0.99))

    return (enc_coor, dec_coor, enc_digit, dec_digit), optimizer, AT


def Save_models(models, PATH):
    (enc_coor, dec_coor, enc_digit, dec_digit) = models
    torch.save(enc_coor.state_dict(), '../weights/enc-coor-' + PATH)
    torch.save(dec_coor.state_dict(), '../weights/dec-coor-' + PATH)
    torch.save(enc_digit.state_dict(), '../weights/enc-digit-' + PATH)
    torch.save(dec_digit.state_dict(), '../weights/dec-digit-' + PATH)
