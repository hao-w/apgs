import sys
sys.path.append("../")

import torch
import probtorch
print('probtorch:', probtorch.__version__, 
      'torch:', torch.__version__, 
      'cuda:', torch.cuda.is_available())

from lstm_model import *
from data_gmm import *
from plots import *
from training import *
from eubo import *

from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torch.distributions.gamma import Gamma
from normal_gamma_conjugacy import *

#Load data
K = 3 ## number of clusters
BATCH_SIZE = 10
data_path = "../gmm_dataset_c2k"
data = torch.from_numpy(np.load(data_path + '/obs.npy')).float()
_, N, D = data.shape
# packed_data = packSeq(data, [N]*data.shape[0], batch_size=B)

#Model params
NUM_HIDDEN_GLOBAL = 16
NUM_HIDDEN_LOCAL = 16
NUM_LAYERS = 1
NUM_SAMPLES = 5

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:1')
PATH = 'oneshot-lstm'

#Training params
NUM_EPISODES = 2000
LEARNING_RATE = 1e-4

enc_eta, enc_z, optimizer = initialize(K, D, 
                                       BATCH_SIZE, NUM_SAMPLES,
                                       NUM_HIDDEN_GLOBAL, NUM_HIDDEN_LOCAL, NUM_LAYERS, 
                                       CUDA, DEVICE, LEARNING_RATE)
train(Eubo_os, enc_eta, enc_z, optimizer, data, K, 
      NUM_EPISODES, NUM_SAMPLES, BATCH_SIZE, PATH, CUDA, DEVICE)
