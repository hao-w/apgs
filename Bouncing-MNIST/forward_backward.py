import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
import probtorch
import math
from torch import logsumexp

def Init_what(crop, os_digit, f_coor, dec_digit, frames, S, training=True, DP=28):
    """
    one-shot predicts z_what
    frames : B * T * 64 * 64
    S : sample_size
    DP : digit height/width
    FP : frame height/width
    """
    B, T, FP, _ = frames.shape
    q_z_what, p_z_what = os_digit(frames.view(B, T, FP*FP), S)
    digit_images = dec_digit(z_what).detach() # stop gradient for convolution
    z_where = f_coor(digit_images)
    # recon_mean, log_recon = dec_digit(frames, z_what, z_where, crop, DP=DP, FP=FP)
    return


def Update_z_where(crop, frames, z_what):
    """
    update the z_where given the frames and digit images
    z_what  S * B * H* W
    """
    crop.frame_to_digit(frames, z_where)

    return
