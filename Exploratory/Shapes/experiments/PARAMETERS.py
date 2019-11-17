## Load dataset
import torch
import numpy as np

data_dir = "/home/hao/Research/apg_data/"
## Load dataset
K = 4
N_c, N_s, N_t = 70, 48, 48
N = N_c+N_s+N_t
Data_squares = []
Data_triangles = []

squares_data_path = data_dir + "ncmm/squares_%d/" % (N_s*K)
data_squares = torch.from_numpy(np.load(squares_data_path + 'ob.npy')).float()
Data_squares.append(data_squares)

triangles_data_path = data_dir + "ncmm/triangles_%d/" % (N_t*K)
data_triangles = torch.from_numpy(np.load(triangles_data_path + 'ob.npy')).float()
Data_triangles.append(data_triangles)
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
LEARNING_RATE =  5 * 1e-4
Train_Params = (NUM_EPOCHS, K, D, SAMPLE_SIZE, BATCH_SIZE)