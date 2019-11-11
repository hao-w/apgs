import numpy as np
import torch
## Model Parameters
NUM_HIDDEN_LOCAL = 32
K = 3 # number of clusters
N = 60 # number of data points
D = 2 # data point dimensions
model_params = (K, D, NUM_HIDDEN_LOCAL)

## training parameters
NAME = 'APG'
APG_SWEEPS = 2
NUM_EPOCHS = 1
SAMPLE_SIZE = 10
BATCH_SIZE = 40
LR = 5 * 1e-4
CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:1')

MODEL_VERSION = 'gmm-%s-%dsweeps-%dsamples-%.4flr' % (NAME, APG_SWEEPS, SAMPLE_SIZE, LR)
print('inference method:%s, apg sweeps:%s, epochs:%s, sample size:%s, batch size:%s, learning rate:%s, cuda:%s, device:%s' % (NAME, APG_SWEEPS, NUM_EPOCHS, SAMPLE_SIZE, BATCH_SIZE, LR, CUDA, DEVICE))

# print('inference method:', NAME,
#       '\tapg sweeps:', APG_SWEEPS,
#       '\tepochs:', NUM_EPOCHS,
#       '\tsample size:', SAMPLE_SIZE,
#       '\tbatch size:', BATCH_SIZE,
#       '\tlearning rate:', LR,
#       '\tcuda:', CUDA,
#       '\tdevice:', DEVICE)
