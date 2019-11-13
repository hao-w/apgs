import numpy as np
import torch
# training parameters
K = 3 # number of digits
T = 10 # timesteps
NUM_EPOCHS = 500
APG_SWEEPS = 10
SAMPLE_SIZE = 10
BATCH_SIZE = 20
LR = 1e-4

## Model Parameters
FRAME_PIXELS = 96
DIGIT_PIXELS = 28
NUM_HIDDEN_DIGIT = 400
NUM_HIDDEN_COOR = 400
Z_WHERE_DIM = 2 # z_where dims
Z_WHAT_DIM = 10

MODEL_NAME = 'apg'
MODEL_VERSION = 'bmnist-%s-%dsweeps-%dsamples-%.4flr' % (MODEL_NAME, APG_SWEEPS, SAMPLE_SIZE, LR)
# LOAD_VERSION = 'bmnist-apg-10sweeps-10samples-0.0005lr'
print('inference method:%s, apg sweeps:%s, epochs:%s, sample size:%s, batch size:%s, learning rate:%s' % (MODEL_NAME, APG_SWEEPS, NUM_EPOCHS, SAMPLE_SIZE, BATCH_SIZE, LR))