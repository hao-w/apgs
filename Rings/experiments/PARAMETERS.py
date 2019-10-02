## Load dataset
import torch
import numpy as np

data_dir = "/home/hao/Research/apg_data/"
## Load dataset
K = 4
N = 50
data_path = data_dir + "ncmm/rings_%d" % N*K
Data = []
data = torch.from_numpy(np.load(data_path + '/ob_%d.npy' % (N * K))).float()
Data.append(data)
## Train Parameters
NUM_EPOCHS = 500
D = 2
SAMPLE_SIZE = 10
BATCH_SIZE = 20
RECON_SIGMA = torch.ones(1) * 0.2
## MOdel Parameters
NUM_HIDDEN_GLOBAL = 32
NUM_HIDDEN_STATE = 32
NUM_HIDDEN_ANGLE = 32
NUM_HIDDEN_DEC = 32
NUM_NSS = 8
HIDDEN_LIST = (NUM_HIDDEN_GLOBAL, NUM_HIDDEN_STATE, NUM_HIDDEN_ANGLE, NUM_HIDDEN_DEC, NUM_NSS)
LEARNING_RATE = 1e-4
Train_Params = (NUM_EPOCHS, K, D, SAMPLE_SIZE, BATCH_SIZE)