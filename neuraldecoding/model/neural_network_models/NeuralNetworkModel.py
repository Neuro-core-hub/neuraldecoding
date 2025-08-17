from typing import Any
from .. import Model
import torch
import numpy as np

class NeuralNetworkModel(Model):
    def __call__(self, data: Any) -> Any:
        pass

    def _train_one_epoch(self, train_data, model, optimizer, loss_func, device):
        pass
    
    def _validate_one_epoch(self, valid_data, model, loss_func, device):
        pass

    def forward(self, input: Any) -> Any:
        pass

    def save_model(self, filepath: str) -> None:
        pass
    
    def load_model(self, filepath: str) -> None:
        pass
