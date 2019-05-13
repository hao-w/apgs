#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.append("../../")
import import_envs
import numpy as np
import torch
import probtorch
from training import *


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('run', '../../import_envs.py')
# print('probtorch:', probtorch.__version__, 
#       'torch:', torch.__version__, 
#       'cuda:', torch.cuda.is_available())


# In[2]:


## Load dataset
data_path = "../gmm_dataset_c5k"
Data = torch.from_numpy(np.load(data_path + '/obs.npy')).float()

NUM_DATASETS, N, D = Data.shape
K = 3 ## number of clusters
SAMPLE_SIZE = 10
NUM_HIDDEN_LOCAL = 32
NUM_HIDDEN_GLOBAL = 32
NUM_LAYERS = 32

BATCH_SIZE = 20
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
CUDA = torch.cuda.is_available()
PATH = 'baseline-lstm-%dsamples' % (SAMPLE_SIZE)
DEVICE = torch.device('cuda:0')

Train_Params = (NUM_EPOCHS, NUM_DATASETS, SAMPLE_SIZE, BATCH_SIZE, CUDA, DEVICE, PATH)
Model_Params = (N, K, D)


# In[3]:


from local_enc import *
from global_lstm import *
## if reparameterize continuous variables
Reparameterized = False
# initialization
enc_z = Enc_z(K, D, NUM_HIDDEN_LOCAL, CUDA, DEVICE)
oneshot_eta = LSTM_eta(K, D, BATCH_SIZE, SAMPLE_SIZE, NUM_HIDDEN_GLOBAL, NUM_LAYERS, CUDA, DEVICE, Reparameterized)
if CUDA:
    enc_z.cuda().to(DEVICE)
if CUDA:
    oneshot_eta.cuda().to(DEVICE)
optimizer =  torch.optim.Adam(list(oneshot_eta.parameters())+list(enc_z.parameters()),lr=LEARNING_RATE, betas=(0.9, 0.99))
models = (oneshot_eta, enc_z)


# In[4]:


from os_ep import *
train(models, EP, optimizer, Data, Model_Params, Train_Params)


# In[5]:


torch.save(enc_z.state_dict(), '../weights/enc-z-%s' + PATH)
torch.save(oneshot_eta.state_dict(), '../weights/oneshot-eta-%s' + PATH)


# In[12]:


def test(models, objective, Data, Model_Params, Train_Params):
    (NUM_EPOCHS, NUM_DATASETS, S, B, CUDA, device, path) = Train_Params
    SubTrain_Params = (device, S, B) + Model_Params
    ##(N, K, D, mcmc_size) = Model_Params
    indices = torch.randperm(NUM_DATASETS)
    batch_indices = indices[0*B : (0+1)*B]
    obs = Data[batch_indices]
    obs = shuffler(obs).repeat(S, 1, 1, 1)
    if CUDA:
        obs =obs.cuda().to(device)
    loss, metric_step, reused = objective(models, obs, SubTrain_Params)
    return obs, metric_step, reused


# In[13]:


BATCH_SIZE_TEST = 50
Train_Params_Test = (NUM_EPOCHS, NUM_DATASETS, SAMPLE_SIZE, BATCH_SIZE_TEST, CUDA, DEVICE, PATH)
obs, metric_step, reused = test(models, EP, Data, Model_Params, Train_Params_Test)
(q_eta, _, q_z, _, _, _) = reused


# In[14]:


get_ipython().run_line_magic('time', 'plot_samples(obs, q_eta, q_z, PATH)')


# In[ ]:


incremental_gap = symkl_test.cpu().data.numpy()[1:]
M = incremental_gap.shape[0]
overall_gap = np.zeros(M)
for m in range(M):
    overall_gap[m] = incremental_gap[:m+1].sum()


# In[ ]:


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
plt.yscale("log")
ax.plot(incremental_gap, label="incremental gap")
ax.plot(overall_gap, label='overall gap')
ax.legend(fontsize=14)
ax.set_xlabel('Steps')


# In[ ]:


incremental_gap

