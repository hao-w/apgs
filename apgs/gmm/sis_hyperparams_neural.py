# # training parameters
# NUM_EPOCHS = 50
# # APG_SWEEPS = 1
# # SAMPLE_SIZE = 25
# BATCH_SIZE = 20
# LR =  5 * 1e-4
# ## Model Parameters
# MODEL_NAME = 'sis' # options: apg | baseline-mlp | baesline-lstm
# K = 3 # number of clusters
# D = 2 # data point dimensions
# NUM_HIDDEN_APG_ETA = 32

# NUM_HIDDEN_APG_Z = 32
# model_params = (K, D, NUM_HIDDEN_APG_Z)
# # MODEL_VERSION = 'gmm-%s-%dsweeps-%dsamples-%.4flr' % (MODEL_NAME, APG_SWEEPS, SAMPLE_SIZE, LR)
# # print('inference method:%s, apg sweeps:%s, epochs:%s, sample size:%s, batch size:%s, learning rate:%s' % (MODEL_NAME, APG_SWEEPS, NUM_EPOCHS, SAMPLE_SIZE, BATCH_SIZE, LR))

# training parameters
NUM_EPOCHS = 50
# APG_SWEEPS = 4
# SAMPLE_SIZE = 10
BATCH_SIZE = 20
LR = 5 * 1e-4
## Model Parameters
MODEL_NAME = 'apg' # options: apg | baseline-mlp | baesline-lstm

K = 3 # number of clusters
D = 2 # data point dimensions
NUM_HIDDEN_APG_ETA = 32
NUM_HIDDEN_APG_Z = 32
model_params = (K, D, NUM_HIDDEN_APG_ETA, NUM_HIDDEN_APG_Z)