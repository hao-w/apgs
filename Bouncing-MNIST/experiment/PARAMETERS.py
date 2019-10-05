import numpy as np
import torch

data_dir = "/home/hao/Research/apg_data/"

K = 2
FRAME_PIXELS = 64
DIGIT_PIXELS = 28


data_path = '/data/hao/apg_data/bmnist_2digits/'
mnist_mean = torch.from_numpy(np.load('../mnist_mean.npy')).float()
NUM_GROUPS = 60

## Train Parameters
NUM_EPOCHS = 500
T = 10
D = 2
SAMPLE_SIZE = 10
BATCH_SIZE = 20
## MOdel Parameters
NUM_HIDDEN_DIGIT = 400
NUM_HIDDEN_COOR = 400
Z_WHAT_DIM = 10
HIDDEN_LIST = (NUM_HIDDEN_DIGIT, NUM_HIDDEN_COOR, Z_WHAT_DIM)
Sigma0 = 1
LEARNING_RATE = 1e-4
Train_Params = (NUM_EPOCHS, NUM_GROUPS, SAMPLE_SIZE, BATCH_SIZE)
