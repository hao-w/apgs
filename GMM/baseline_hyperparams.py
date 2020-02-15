# training parameters
NUM_EPOCHS = 300
SAMPLE_SIZE = 100
BATCH_SIZE = 20
LR =  1e-4
## Model Parameters
K = 3 # number of clusters
D = 2 # data point dimensions
NUM_HIDDEN_Z = 32
model_params = (K, D, NUM_HIDDEN_Z)
MODEL_NAME = 'baseline-%s' % ARCHITECTURE # options: baseline-mlp | baesline-lstm
MODEL_VERSION = 'gmm-%s-%dsamples-%.4flr' % (MODEL_NAME, SAMPLE_SIZE, LR)
LOAD_VERSION = 'apg-new-systematic-9sweeps-5samples'
print('epochs:%s, sample size:%s, batch size:%s, learning rate:%s' % (NUM_EPOCHS, SAMPLE_SIZE, BATCH_SIZE, LR))
