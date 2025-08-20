from typing import Any
from neuraldecoding.model.Model import Model
import torch
import numpy as np

class NeuralNetworkModel(Model):
    def __call__(self, data):
        return self.forward(data)
    
    def forward(self, input: Any) -> Any:
        pass

    def train_step(self, x, y, optimizer, loss_func, clear_cache = False):
        yhat = self.forward(x)

        loss = loss_func(yhat, y)

        loss.backward()
        optimizer.step()
        if(clear_cache):
            del x, y

        return loss, yhat

    def save_model(self, filepath: str) -> None:
        pass
    
    def load_model(self, filepath: str) -> None:
        pass
