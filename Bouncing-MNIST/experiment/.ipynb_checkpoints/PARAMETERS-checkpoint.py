import numpy as np
import torch

K = 3
FRAME_PIXELS = 96
DIGIT_PIXELS = 28


data_path = "/data/hao/apg_data/mnist/train2/bmnist_3digits/"
#data_path_test = "/home/hao/Research/apg_data/mnist/test/bmnist_3digits/"

mnist_mean = torch.from_numpy(np.load('../mnist_mean.npy')).float()
NUM_GROUPS = 60
#NUM_GROUPS_TEST = 6
## Train Parameters
NUM_EPOCHS = 500
T = 10
#T_test =100
D = 2
SAMPLE_SIZE = 20
BATCH_SIZE = 5
## MOdel Parameters
NUM_HIDDEN_DIGIT = 400
NUM_HIDDEN_COOR = 400
Z_WHAT_DIM = 10
HIDDEN_LIST = (NUM_HIDDEN_DIGIT, NUM_HIDDEN_COOR, Z_WHAT_DIM)
LEARNING_RATE = 1e-4
Train_Params = (NUM_EPOCHS, NUM_GROUPS, SAMPLE_SIZE, BATCH_SIZE)
