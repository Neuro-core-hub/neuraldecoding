import torch
import torch.nn as nn
from neuraldecoding.model.neural_network_models.NeuralNetworkModel import NeuralNetworkModel
import numpy as np
import os

class LSTM(nn.Module, NeuralNetworkModel):
    def __init__(self, params):
        ''' 
        Initializes a LSTM

         Args:
            params:                dict containing the following model params:
                                            input_size:         number of input features
                                            num_outputs:        number of output features
                                            hidden_size:        size of hidden state in model 
                                            num_layers:         number of layers in model
                                            rnn_type:           specifies what type of recurrent model (gru, lstm, rnn)
                                            device:             optional, specifies what device to compute on. Default is cpu.
                                            hidden_noise_std:   optional, deviation of hidden noise to add to model
                                            dropout_input:      optional, drops out some input during forward pass
                                            drop_prob:          optional, specifies probability of layer of model being dropped
        Returns:
            None
        ''' 

        params["rnn_type"] = "lstm"
        # nn.Module.__init__(self)
        # NeuralNetworkModel.__init__(self, model_params)

        super(LSTM, self).__init__()

        self.input_size = params["input_size"]
        self.hidden_size = params["hidden_size"]
        self.num_layers = params["num_layers"]
        self.num_outputs = params["num_outputs"]
        self.device = params.get("device", "cpu") 
        self.hidden_noise_std = params.get("hidden_noise_std", 0.0)
        self.dropout_input = params.get("dropout_input", False)
        self.drop_prob = params.get("drop_prob", 0.0)

        # Define LSTM layer
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.drop_prob if self.num_layers > 1 else 0  # Dropout only applies if num_layers > 1
        )

        # Define output layer
        self.fc = nn.Linear(self.hidden_size, self.num_outputs)

        # Dropout layer for input (if enabled)
        self.input_dropout = nn.Dropout(self.drop_prob) if self.dropout_input else nn.Identity()

    def forward(self, x, h=None, return_all_tsteps=False, return_h = False):
        """
        Runs forward pass of LSTM Model

        Args:
            x:                  Neural data tensor of shape (batch_size, num_inputs, sequence_length) or (batch_size, num_inputs)
            h:                  Hidden state tensor of shape (n_layers, batch_size, hidden_size) [for LSTM, its a tuple of two of these, one for hidden state, one for cell state]
            return_all_steps:   If true, returns predictions from all timesteps in the sequence. If false, only returns the
                                last step in the sequence.
        Returns:
            out:                output/prediction from forward pass of shape (batch_size, seq_len^, num_outs)  ^if return_all_steps is true
            h:                  Hidden state tensor of shape (n_layers, batch_size, hidden_size) [for LSTM, its a tuple of two of these, one for hidden state, one for cell state]
        """

        if x.dim() != 3 and x.dim() != 2:
            raise ValueError(f"Input tensor must be 3D (batch_size, num_inputs, sequence_length) or 2D, got shape {x.shape}")
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input feature dimension mismatch: expected {self.input_size}, got {x.shape[1]}")
        
        if x.dim() == 3:
            x = x.permute(0, 2, 1)  # put in format (batches, sequence length (history), features)
            if h is None:
                h = self.init_hidden(x.shape[0], dim=3) # x.shape[0] is batch size
        elif x.dim() == 2:
            if h is None:
                h = self.init_hidden(x.shape[0], dim=2)
        else:
            raise ValueError(f"Invalid input tensor dimension {x.dim()}. Expected 2 or 3 dimensions.")
        
        if self.dropout_input and self.training:
            x = self.input_dropout(x)
            
        out, h = self.rnn(x, h) # out shape:    (batch_size, seq_len, hidden_size) like (64, 20, 350)
                                # h shape:      (n_layers, batch_size, hidden_size) like (2, 64, 350)

        if return_all_tsteps:
            out = self.fc(out)  # out now has shape (batch_size, seq_len, num_outs) like (64, 20, 2)
        else:
            if x.dim() == 2:  # unbatched input
                out = self.fc(out[-1])  # shape: (hidden_size,) -> (num_outputs,)
            else:  # batched input
                out = self.fc(out[:, -1]) # out now has shape (batch_size, num_outs) like (64, 2)
        if return_h:
            return out, h
        else:
            return out
    

    def init_hidden(self, batch_size, dim=3):
        """
        Initializes hidden state of LSTM Model

        Args:
            batch_size:   integer describing current batch size

        Returns:
            hidden:       hidden state tensor of shape (n_layers, batch_size, hidden_size) [for LSTM, its a tuple of two of these, one for hidden state, one for cell state]
        """
        # lstm - create a tuple of two hidden states
        if dim == 3:
            if self.hidden_noise_std:
                hidden = (torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                        std=self.hidden_noise_std).to(device=self.device),
                            torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                        std=self.hidden_noise_std).to(device=self.device))
            else:
                hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device),
                            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device))
        elif dim == 2:
            if self.hidden_noise_std:
                hidden = (torch.normal(mean=torch.zeros(self.num_layers, self.hidden_size),
                                        std=self.hidden_noise_std).to(device=self.device),
                            torch.normal(mean=torch.zeros(self.num_layers, self.hidden_size),
                                        std=self.hidden_noise_std).to(device=self.device))
            else:
                hidden = (torch.zeros(self.num_layers, self.hidden_size).to(device=self.device),
                            torch.zeros(self.num_layers, self.hidden_size).to(device=self.device))
        else:
            raise ValueError(f"Invalid dimension {dim} for hidden state initialization. Expected 2 or 3.")
        return hidden

    def save_model(self, filepath):
        """
        Saves the model in its current state at the specified filepath

        Parameters:
            filepath (path-like object) indicates the file path to save the model in

        """
        checkpoint_dict = {
            "model_state_dict": self.state_dict(),
            "model_params": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            "model_type": "LSTM"
        }
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(checkpoint_dict, filepath)


    def load_model(self, filepath):
        """
        Load model parameters from a specified location

        Parameters:
            filepath (path-like object) indicates the file path to load the model from
        """
        
        checkpoint = torch.load(filepath)

        if checkpoint["model_type"] != "LSTM":
            raise Exception("Tried to load model that isn't a LSTM Instance")
        
        self.load_state_dict(checkpoint["model_state_dict"])

        model_params = checkpoint["model_params"]
        self.hidden_size = model_params["hidden_size"]
        self.num_layers = model_params["num_layers"]

