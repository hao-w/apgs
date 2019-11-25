import numpy as np
import torch
import os
# training parameters
K = 3 # number of digits
T = 10 # timesteps
NUM_EPOCHS = 500
# APG_SWEEPS = 6
SAMPLE_SIZE = 10
BATCH_SIZE = 10
LR = 1e-4

## Model Parameters
FRAME_PIXELS = 96
DIGIT_PIXELS = 28
NUM_HIDDEN_DIGIT = 400
NUM_HIDDEN_COOR = 400
Z_WHERE_DIM = 2 # z_where dims
Z_WHAT_DIM = 10

MODEL_NAME = 'vae'
MODEL_VERSION = 'bmnist-%ddigits-%s-%dsamples-%.4flr' % (K, MODEL_NAME, SAMPLE_SIZE, LR)
LOAD_VERSION = 'bmnist-3digits-apg-10samples-0.0001lr'
# LOAD_VERSION_GALILEO = 'bmnist-3digits-apg-10samples-0.0001lr-galileo'
print('inference method:%s, epochs:%s, sample size:%s, batch size:%s, learning rate:%s' % (MODEL_NAME, NUM_EPOCHS, SAMPLE_SIZE, BATCH_SIZE, LR))

MNIST_MEAN_PATH = '../mnist_mean.npy'
