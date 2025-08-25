import torch
import torch.nn as nn
import torch.nn.functional as F
from .NeuralNetworkModel import NeuralNetworkModel
import numpy as np
import os

class RecurrentModel(nn.Module, NeuralNetworkModel):
    ''' A general recurrent model that can use VanillaRNN/GRU/LSTM, with a linear layer to the output '''
    def __init__(self, params):
        '''
        params contains:
        - input_size
        - hidden_size
        - num_outputs
        - num_layers
        - rnn_type: default 'lstm'
        - drop_prob: default 0
        - hidden_noise_std: default None
        - dropout_input: default 0
        '''
        super().__init__()
        
        self.model_params = params

        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.num_outputs = params['num_outputs']
        self.num_layers = params['num_layers']
        self.rnn_type = params.get('rnn_type', 'lstm').lower()
        self.hidden_noise_std = params.get('hidden_noise_std', None)
        drop_prob = params.get('dropout', 0)
        dropout_input = params.get('dropout_input', 0)

        if dropout_input:
            self.dropout_input = nn.Dropout(dropout_input)
        else:
            self.dropout_input = None

        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, dropout=drop_prob, batch_first=True, nonlinearity='relu')
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=drop_prob, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_outputs)

    def forward(self, x, h=None,  return_all_tsteps=False, return_h=False):
        """
        x:                  Neural data tensor of shape (batch_size, num_inputs, sequence_length)
        h:                  Hidden state tensor
        return_all_steps:   If true, returns predictions from all timesteps in the sequence. If false, only returns the
                            last step in the sequence.
        """
        x = x.permute(0, 2, 1)  # put in format (batches, sequence length (history), features)

        if self.dropout_input and self.training:
            x = self.dropout_input(x)

        if h is None:
            h = self.init_hidden(x.shape[0])

        out, h = self.rnn(x, h)
        # out shape:    (batch_size, seq_len, hidden_size) like (64, 20, 350)
        # h shape:      (n_layers, batch_size, hidden_size) like (2, 64, 350)

        if return_all_tsteps:
            out = self.fc(out)  # out now has shape (batch_size, seq_len, num_outs) like (64, 20, 2)
        else:
            out = self.fc(out[:, -1])  # out now has shape (batch_size, num_outs) like (64, 2)

        if return_h:
            return out, h
        else:
            return out

    def init_hidden(self, batch_size):
        if self.rnn_type == 'lstm':
            # lstm - create a tuple of two hidden states
            if self.hidden_noise_std:
                hidden = (torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                       std=self.hidden_noise_std),
                          torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                       std=self.hidden_noise_std))
            else:
                hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                          torch.zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            # not an lstm - just a single hidden state vector
            if self.hidden_noise_std:
                hidden = torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                      std=self.hidden_noise_std)
            else:
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden
    
    def save_model(self, filepath):
        checkpoint_dict = {
            "model_state_dict": self.state_dict(),
            "model_params": self.model_params,
            "model_type": "GenericRNN"
        }
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(checkpoint_dict, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)

        if checkpoint["model_type"] != "GenericRNN":
            raise Exception("Tried to load model that isn't a GenericRNN Instance")
        
        if self.model_params != checkpoint["model_params"]:
            raise ValueError("Model parameters do not match the checkpoint parameters")

        self.load_state_dict(checkpoint["model_state_dict"])

        self.model_params = checkpoint["model_params"]