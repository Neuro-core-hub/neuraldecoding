import torch
import torch.nn as nn
import torch.nn.functional as F
from neuraldecoding.model.neural_network_models.NeuralNetworkModel import NeuralNetworkModel
import numpy as np
import os

class Adaline(nn.Module, NeuralNetworkModel):
    ''' A single-layer nn model '''
    def __init__(self, params):
        '''
        params contains:
        - input_size
        - num_outputs
        - history
        - device
        - append_ones
        '''
        super().__init__()
        
        self.model_params = params

        self.input_size = params['input_size']
        self.num_outputs = params['num_outputs']
        self.history = params['history']
        self.device = params.get("device", "cpu")
        self.append_ones = params.get("append_ones", False)

        if self.append_ones:
            self.linear = nn.Linear(self.input_size*self.history+1, self.num_outputs, bias=False) # add one for bias term
        else:
            self.linear = nn.Linear(self.input_size*self.history, self.num_outputs)

    def forward(self, x):
        """
        x:                  Neural data tensor of shape (num_inputs*history, num_outputs)
        """
        
        out = self.linear(x)
        return out 
    
    def save_model(self, filepath):
        checkpoint_dict = {
            "model_state_dict": self.state_dict(),
            "model_params": self.model_params,
            "neural_scaler": getattr(self, 'neural_scaler', None),
            "behavior_scaler": getattr(self, 'behavior_scaler', None),
            "model_type": "Adaline"
        }
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(checkpoint_dict, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)

        if checkpoint["model_type"] != "Adaline":
            raise Exception("Tried to load model that isn't an Adaline Instance")
        
        if self.model_params != checkpoint["model_params"]:
            raise ValueError("Model parameters do not match the checkpoint parameters")

        self.load_state_dict(checkpoint["model_state_dict"])

        self.model_params = checkpoint["model_params"]
        if "neural_scaler" in checkpoint:
            self.neural_scaler = checkpoint["neural_scaler"]
        else:
            self.neural_scaler = None
        if "behavior_scaler" in checkpoint:
            self.behavior_scaler = checkpoint["behavior_scaler"]
        else:
            self.behavior_scaler = None

class AdalineTrialInput(Adaline):
    def __init__(self, params):
        """
        Initializes an Adaline with trial input support

        Args:
            model_params:                dict containing the same parameters as Adaline
        """
        super(AdalineTrialInput, self).__init__(params)

    def train_step(self, x, y, optimizer, loss_func, clear_cache = False, return_y = False):
        if x.dim() >= 3:
            x = x.squeeze(0)  # Remove extra batch dimension for trial input models
        if y.dim() >= 3:
            y = y.squeeze(0)  # Remove extra batch dimension for trial input models

        yhat = self.forward(x)

        loss = loss_func(yhat, y)

        loss.backward()
        optimizer.step()
        if(clear_cache):
            del x, y

        if return_y:
            return loss, yhat, y
        else:
            return loss, yhat
