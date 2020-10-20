# training parameters
NUM_EPOCHS = 300
APG_SWEEPS = 10
SAMPLE_SIZE = 10
BATCH_SIZE = 20
LR =  5 * 1e-4

MODEL_NAME = 'sis' # options: apg | baseline-mlp | baesline-lstm
K = 4
D = 2
NUM_HIDDEN_GLOBAL = 32
NUM_HIDDEN_STATE = 32
NUM_HIDDEN_ANGLE = 32
NUM_HIDDEN_DEC = 32
NUM_NSS = 8
RECON_SIGMA =  0.3
model_params = (K, D, NUM_HIDDEN_GLOBAL, NUM_NSS, NUM_HIDDEN_STATE, NUM_HIDDEN_ANGLE, NUM_HIDDEN_DEC, RECON_SIGMA)
MODEL_VERSION = 'dgmm-%s-%dsweeps-%dsamples-%.4flr' % (MODEL_NAME, APG_SWEEPS, SAMPLE_SIZE, LR)
LOAD_VERSION = 'dgmm-sis-10sweeps-10samples-0.0005lr'
print('inference method:%s, apg sweeps:%s, epochs:%s, sample size:%s, batch size:%s, learning rate:%s' % (MODEL_NAME, APG_SWEEPS, NUM_EPOCHS, SAMPLE_SIZE, BATCH_SIZE, LR))
