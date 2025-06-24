import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from neuraldecoding.model.neural_network_models.NeuralNetworkModel import NeuralNetworkModel
from neuraldecoding.model.neural_network_models.LSTM import LSTM




def create_model(self, model_config: DictConfig) -> torch.nn.Module:
    """Creates and returns a loss function based on the configuration."""
    model_class = globals()[model_config['type']]  
    model = model_class(model_config['parameters']) 
    return model
