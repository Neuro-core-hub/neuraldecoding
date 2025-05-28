import numpy as np
import pickle
import torch
from typing import Any
from .LinearModel import LinearModel
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import Ridge

class Regression(LinearModel):
    def __init__(self, params = {}):
        super().__init__(params)

    def __call__(self, data: Any, is_torch = False) -> Any:
        return self.forward(data, is_torch)

    def train_step(self, input_data: Any) -> None:
        X, y = input_data
        self.model.fit(X, y)

    def forward(self, data: Any, is_torch = False) -> Any:
        predictions = self.model.predict(data)
        if is_torch:
            return torch.tensor(predictions, dtype=torch.float64)
        else:
            return predictions

    def save_model(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

class LinearRegression(Regression):
    def __init__(self, params = {}):
        super().__init__(params)
        self.model = SklearnLinearRegression(**params)

class RidgeRegression(Regression):
    def __init__(self, params = {}):
        super().__init__(params)
        self.model = Ridge(**params)
