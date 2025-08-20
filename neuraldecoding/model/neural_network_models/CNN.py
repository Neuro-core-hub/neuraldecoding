import torch
import torch.nn as nn
import torch.nn.functional as F
from neuraldecoding.model.neural_network_models.NeuralNetworkModel import NeuralNetworkModel
import numpy as np
import os
import warnings
warnings.warn("CNN model not implemented yet")
class CNNModel(nn.Module, NeuralNetworkModel):
    def __init__(self, params):
        super().__init__()
        self.model_params = params
        num_inputs = params['num_inputs']
        num_outputs = params['num_outputs']
        # i dont know what to do with it. what scenario is a cnn needed?