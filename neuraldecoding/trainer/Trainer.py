from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import torch

from neuraldecoding.model.Model import Model

class Trainer(ABC):
    @abstractmethod
    def train_model(self, 
                    train_data: Union[np.ndarray, torch.Tensor],
                    valid_data: Union[np.ndarray, torch.Tensor],
                    test_data: Union[np.ndarray, torch.Tensor],
                    model: Model,
                    params: List) -> Model:
        """
        Trains the model using the provided training, validation, and test data.
        
        Parameters:
            train_data: Training dataset in the form of numpy or torch tensors.
            valid_data: Validation dataset in the form of numpy or torch tensors.
            test_data: Test dataset in the form of numpy or torch tensors.
            model: An instance of Model to be trained.
            params: A list of parameters for training.
        
        Returns:
            A trained Model instance.
        """
        pass