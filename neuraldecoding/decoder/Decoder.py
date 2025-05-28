import copy
from abc import ABC, abstractmethod

import numpy as np
import torch

from neuraldecoding.model.linear_models import KalmanFilter, LinearRegression, RidgeRegression
from neuraldecoding.model.neural_network_models import LSTM

from neuraldecoding.stabilization.latent_space_alignment import LatentSpaceAlignment

model_reg = {
    "KalmanFilter": KalmanFilter,
    "LinearRegression": LinearRegression,
    "RidgeRegression": RidgeRegression,
    "LSTM": LSTM
    }

stabilization_reg = {
    "LatentSpaceAlignment": LatentSpaceAlignment
    }

class Decoder(ABC):
    def __init__(self, cfg):
        """
        Decoder Class

        Args:
            cfg: config
        """
        # Stabilization implementation, not verified commented out for now
        # # Get stabilization stuff
        # if cfg["stabilization"]["name"] in model_reg:
        #     self.stabilization = stabilization_reg[cfg["stabilization"]["name"]](**cfg["stabilization"]["parameters"])
        # else:
        #     raise ValueError(f"Model {cfg['stabilization']['name']} is not registered in stabilization_reg.")

        # Get model stuff
        if cfg["model"]["name"] in model_reg:
            self.model = model_reg[cfg["model"]["name"]](cfg["model"]["parameters"])
        else:
            raise ValueError(f"Model {cfg['model']['name']} is not registered in model_reg.")
        
        # Get model path
        self.fpath = cfg["fpath"]

        # Get model i/o shape
        self.input_shape = cfg["model"]["input_shape"]
        self.output_shape = cfg["model"]["output_shape"]

        # Load model from path
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
    
class NeuralNetworkDecoder(Decoder):
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