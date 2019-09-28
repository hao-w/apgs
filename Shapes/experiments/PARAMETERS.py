## Load dataset
import torch
import numpy as np

data_dir = "/home/hao/Research/apg_data/"
## Load dataset
K = 3
N_c, N_s, N_t = 70, 72, 72
N = N_c+N_s+N_t
Data = []
data_path = data_dir + "ncmm/shapes_c=%d_s=%d_t=%d/" % (N_c, N_s, N_t)
data = torch.from_numpy(np.load(data_path + 'ob.npy')).float()
Data.append(data)
## Train Parameters
NUM_EPOCHS = 500
D = 2
SAMPLE_SIZE = 10
BATCH_SIZE = 20
RECON_SIGMA = torch.ones(1) * 0.2
## MOdel Parameters
NUM_HIDDEN_GLOBAL = 32
NUM_HIDDEN_STATE = 64
NUM_HIDDEN_ANGLE = 32
NUM_HIDDEN_DEC = 32
NUM_NSS = 16
HIDDEN_LIST = (NUM_HIDDEN_GLOBAL, NUM_HIDDEN_STATE, NUM_HIDDEN_ANGLE, NUM_HIDDEN_DEC, NUM_NSS)
LEARNING_RATE = 5 * 1e-4
Train_Params = (NUM_EPOCHS, K, D, SAMPLE_SIZE, BATCH_SIZE)