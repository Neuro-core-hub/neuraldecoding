from .Decoder import Decoder
import torch
import numpy as np
from abc import ABC, abstractmethod
class OfflineDecoder(Decoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def predict(self, neural_data):
        """
        Predict outputs given neural data in offline setting.

        Args:
            neural_data (numpy.ndarray): Input neural data of shape [N, numfeats]

        Returns:
            prediction (torch.Tensor): Predicted output
        """
        prediction = self.model(neural_data)
        
        return prediction

class LinearDecoder(OfflineDecoder):
    def __init__(self, cfg):
        super().__init__(cfg)