# training parameters
NUM_EPOCHS = 300
APG_SWEEPS = 4
SAMPLE_SIZE = 10
BATCH_SIZE = 20
LR =  2.5 * 1e-4
## Model Parameters
MODEL_NAME = 'apg' # options: apg | baseline-mlp | baesline-lstm
RESAMPLING_STRATEGY = 'systematic'

K = 3 # number of clusters
D = 2 # data point dimensions
NUM_HIDDEN_APG_Z = 32
model_params = (K, D, NUM_HIDDEN_APG_Z)
MODEL_VERSION = 'gmm-%s-%s-%dsweeps-%dsamples-%.5flr' % (MODEL_NAME, RESAMPLING_STRATEGY, APG_SWEEPS, SAMPLE_SIZE, LR)
LOAD_VERSION = 'gmm-apg-systematic-9sweeps-10samples-0.00025lr'
# LOAD_VERSION = 'test'
print('inference method:%s, resampling strategy:%s, apg sweeps:%s, epochs:%s, sample size:%s, batch size:%s, learning rate:%s' % (MODEL_NAME, RESAMPLING_STRATEGY, APG_SWEEPS, NUM_EPOCHS, SAMPLE_SIZE, BATCH_SIZE, LR))
