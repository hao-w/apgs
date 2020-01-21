ARCHITECTURE = 'mlp'
# training parameters
NUM_EPOCHS = 30
SAMPLE_SIZE = 50
BATCH_SIZE = 10
LR =  5 * 1e-4
## Model Parameters
K = 3 # number of clusters
D = 2 # data point dimensions
NUM_HIDDEN_Z = 32
if ARCHITECTURE == 'lstm':
    NUM_HIDDEN_LSTM = 32
    NUM_LAYERS = 2
    model_params = (K, D, NUM_HIDDEN_Z, NUM_HIDDEN_LSTM, BATCH_SIZE, SAMPLE_SIZE, NUM_LAYERS)
else:
    model_params = (K, D, NUM_HIDDEN_Z)
MODEL_NAME = 'baseline-%s' % ARCHITECTURE # options: baseline-mlp | baesline-lstm
MODEL_VERSION = 'gmm-%s-%dsamples-%.4flr' % (MODEL_NAME, SAMPLE_SIZE, LR)
MODEL_VERSION = 'gmm-%s-%dsamples-%.4flr' % (MODEL_NAME, SAMPLE_SIZE, LR)

print('inference method:%s, epochs:%s, sample size:%s, batch size:%s, learning rate:%s' % (MODEL_NAME, NUM_EPOCHS, SAMPLE_SIZE, BATCH_SIZE, LR))
