import numpy as np
import torch as to
import torch.nn as nn
from util_data import *
      
def genSeq(T_min, T_max, dt, init_v, noise_cov, boundary, num_seq):
    # Generate Sequences
    Seq = []
    Len = []
    for i in range(num_seq):
        T = np.random.randint(T_min, T_max)
        State, Disp, A_true, Zs_true = generate_seq(T, dt, boundary, init_v, noise_cov)
        Seq.append(to.tensor(Zs_true, dtype=torch.float))
        Len.append(T)
    return Seq, Len

def PackSeq(Seq, Len, batch_size):
    batches = []
    for b in range(len(Seq) // batch_size):
        # Sort Sequences
        Seq_sorted = []
        Len_sorted = []
        for l,s in sorted(zip(Len[b*batch_size:(b+1)*batch_size], Seq[b*batch_size:(b+1)*batch_size]),
                          key=lambda pair: -pair[0]):
            Seq_sorted.append(s)
            Len_sorted.append(l)
        Len_sorted = to.tensor(Len_sorted, dtype=torch.long)

        # Pack Sequences
        Seq_padded = to.nn.utils.rnn.pad_sequence(Seq_sorted)
        Seq_packed  = to.nn.utils.rnn.pack_padded_sequence(Seq_padded, Len_sorted)#, batch_first=True)
        batches.append((Seq_packed, Len_sorted))

    return batches
    
    
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, target_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.target_dim = target_dim

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.target_dim)
        self.init_hidden()
        
    def init_hidden(self):
        self.hidden = (to.zeros(1, self.batch_size, self.hidden_dim),
                       to.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, x, lenghts):
        # Input: seq_length x batch_size x input_size (embedding_dimension in this case)
        # Output: seq_length x batch_size x hidden_size
        # last_hidden_state: batch_size, hidden_size
        # last_cell_state: batch_size, hidden_size
        out_packed, self.hidden = self.lstm(x, self.hidden)
        out_unpacked, _ = to.nn.utils.rnn.pad_packed_sequence(out_packed)#, batch_first=True)
        out_last = out_unpacked[lenghts - 1, np.arange(out_unpacked.shape[1]), :]
        
        out_lin = self.linear(out_last)
        out = to.exp(out_lin)
        return out
        
        
def train_epoch(batches, model, optimizer, loss_fn):
    epoch_loss = 0
    N = 1
    model.train()
    for (seq_pack, len_seq) in batches:
        model.zero_grad() 
        model.init_hidden()
        N += len(batches[1])

        fx = model(seq_pack, len_seq)
        batch_loss = loss_fn(fx)
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()

    return epoch_loss / N