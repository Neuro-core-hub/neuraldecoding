import copy
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import torch

from omegaconf import OmegaConf, DictConfig
from neuraldecoding.model.linear_models import KalmanFilter, LinearRegression, RidgeRegression, LDA
from neuraldecoding.model.neural_network_models import LSTM
from neuraldecoding.model.Model import DummyModel
import neuraldecoding.stabilization.latent_space_alignment
from neuraldecoding.stabilization.latent_space_alignment import LatentSpaceAlignment

from neuraldecoding.utils.data_tools import prep_data_decoder

MODEL_REGISTRY = {
    "KalmanFilter": KalmanFilter,
    "LinearRegression": LinearRegression,
    "RidgeRegression": RidgeRegression,
    "LDA":LDA,
    "LSTM": LSTM,
    "dummy": DummyModel
    }

STABILIZATION_REGISTRY = {
    "LatentSpaceAlignment": LatentSpaceAlignment
    }

class Decoder(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Decoder Class

        Args:
            cfg: config dictionary
        """
        # Get model stuff
        self.cfg = cfg
        if self.cfg.model_type in MODEL_REGISTRY:
            self.model = MODEL_REGISTRY[self.cfg.model_type](self.cfg.model_params)
        else:
            raise ValueError(f"Model {cfg.model_type} either does not exist or is not registered in the model registry (in Decoder module).")
        self.device = self.cfg.model_params.get("device", "cpu")

        # Get model path
        self.fpath = self.cfg.get('model_path')

        # # Get model i/o shape
        # self.input_shape = cfg["model"]["params"]["input_size"]
        # self.output_shape = cfg["model"]["params"]["num_outputs"]

        self.input_shape = 0
        self.output_shape = 0

        # Load model from path
        self.load_model()

    def load_model(self, fpath : str = None) -> None:
        """
        Load decoder's model parameters from a specified location

        Args:
            fpath: Override path to the parameters
        """
        self.fpath = fpath if fpath is not None else self.fpath
        self.model.load_model(self.fpath)

    def get_decoder(self) -> Self:
        """
        Returns deep-copied decoder object with parameters loaded from the specified file path.
        """
        dec = copy.deepcopy(self)
        dec.load_model()
        return dec
    
    # def stabilize(self, data_0, data_k):
    #     ls_0 = self.stabilization.train(data_0)
    #     ls_k = self.stabilization.extract_latent_space(data_k)
    #     return ls_0, ls_k
    
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
    
class LinearDecoder(Decoder):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
    def predict(self, neural_data: np.ndarray) -> torch.Tensor:
        """
        Predict outputs given neural data in offline setting.

        Args:
            neural_data (numpy.ndarray): Input neural data of shape [N, numfeats]

        Returns:
            prediction (torch.Tensor): Predicted output
        """
        prediction = self.model(neural_data)
        
        return prediction
    
class NeuralNetworkDecoder(Decoder):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
    def predict(self, input):
        with torch.no_grad():
            input = input.to(self.device)
            prediction = self.model(input)
        return prediction
    
class DummyDecoder(Decoder):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

    def predict(self, neural_data):
        self.model(neural_data)
    
