import copy
import torch
from abc import ABC, abstractmethod

from neuraldecoding.model.linear_models import KalmanFilter, LinearRegression, RidgeRegression
from neuraldecoding.model.neural_network_models import LSTM
import numpy as np

model_reg = {
    "KalmanFilter": KalmanFilter,
    "LinearRegression": LinearRegression,
    "RidgeRegression": RidgeRegression,
    "LSTM": LSTM
    }

class Decoder(ABC):
    def __init__(self, cfg):
        """
        Decoder Class

        Args:
            cfg: config
        """
        if cfg["model"]["name"] in model_reg:
            self.model = model_reg[cfg["model"]["name"]](params=cfg["model"]["parameters"])
        else:
            raise ValueError(f"Model '{cfg["model"]["name"]}' is not registered in model_reg.")
        self.fpath = cfg["fpath"]
        self.input_shape = cfg["model"]["input_shape"]
        self.output_shape = cfg["model"]["output_shape"]
        self.load_model()

    def load_model(self, fpath = None):
        """
        Load decoder's model parameters from a specified location

        Args:
            fpath: Override path to the parameters
        """
        self.fpath = fpath if fpath is not None else self.fpath
        self.model.load_model(self.fpath)

    def get_decoder(self):
        """
        Returns deep-copied decoder object with parameters loaded from the specified file path.
        """
        dec = copy.deepcopy(self)
        dec.load_model()
        return dec

    @abstractmethod
    def predict(self, neural_data) -> torch.tensor:
        """
        Predict outputs given neural data.
        """
        pass

    def get_input_shape(self) -> np.ndarray:
        """
        Returns the input shape of the decoder's model.

        Returns:
            np.ndarray: Input shape of the model
        """
        return np.array(self.input_shape)

    def get_output_shape(self) -> np.ndarray:
        """
        Returns the output shape of the decoder's model.

        Returns:
            np.ndarray: Output shape of the model
        """
        return np.array(self.output_shape)