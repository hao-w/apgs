import numpy as np
import torch
import os
# training parameters
K = 3 # number of digits
T = 10 # timesteps
NUM_EPOCHS = 1000
APG_SWEEPS = 0
SAMPLE_SIZE = 20
BATCH_SIZE = 10
LR =  2* 1e-4

## Model Parameters
FRAME_PIXELS = 96
DIGIT_PIXELS = 28
NUM_HIDDEN_DIGIT = 200
NUM_HIDDEN_COOR = 200
Z_WHERE_DIM = 2 # z_where dims
Z_WHAT_DIM = 10

MODEL_NAME = 'apg-as-baseline'
RESAMPLING_STRATEGY = 'systematic'
MODEL_VERSION = 'bmnist-%ddigits-%s' % (K, MODEL_NAME)
# LOAD_VERSION = 'bmnist-%ddigits-%s' % (K, MODEL_NAME)

LOAD_VERSION = 'bmnist-3digits-apg-systematic-4sweeps-20samples-0.00020lr'
print('inference method:%s, resampling:%s, apg sweeps:%s, epochs:%s, sample size:%s, batch size:%s, learning rate:%s' % (MODEL_NAME, RESAMPLING_STRATEGY, APG_SWEEPS, NUM_EPOCHS, SAMPLE_SIZE, BATCH_SIZE, LR))

MNIST_MEAN_PATH = '../mnist_mean.npy'
