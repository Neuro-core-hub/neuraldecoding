import copy
import os
import pickle
import warnings
from abc import ABC, abstractmethod

from neuraldecoding.model.linear_models.KF import KalmanFilter
from neuraldecoding.model.neural_network_models.LSTM import LSTM

#TODO: add other models if implemented

import torch
import numpy as np

#from utils.data_storing import DataManager

class Decoder(ABC):
    def __init__(self, model, cfg):# DONE: keep stabilization only / change that to generic 'data_transformation' 
        """
        Decoder Class

        Args:
            model: model defined in neuraldecoding/model/
            cfg: config #TODO
        """
        # TODO by default decoder should load the model params
        self.model = model
        self.data_transformation = cfg.data_transformation #TODO: pseudocode here replace with actual cfg implementation

    def load_decoder(self, fpath): #TODO how is path defined? like the one in adaptive alignment or ?
        # change name to load_model.
        """
        Load decoder's model parameters from a specified location

        Args:
            fpath: Path to the parameters
        """
        self.model.load_model(fpath)
        
    def save_decoder(self, fpath): #TODO how is path defined? like the one in adaptive alignment or ?
        """
        Saves the model in its current state at the specified filepath

        Args:
            fpath: Path to the parameters
        """
        dir_path = os.path.dirname(fpath)
    
        if not os.path.exists(dir_path): # Code to make the path if it does not exist. TODO: Remove if 'model' class implements this (not implemented in 'model' class for now).
            os.makedirs(dir_path)
            
        self.model.save_model(fpath)

    def get_decoder(self, fpath):
        """
        Returns deep-copied decoder object with parameters loaded from the specified file path.

        Args:
            fpath: Path to the parameters
        """
        dec = copy.deepcopy(self)
        dec.load_decoder(fpath)
        return dec

    @abstractmethod
    def predict(self, neural_data) -> torch.tensor:
        pass
    
class OfflineDecoder(Decoder):
    def __init__(self, cfg, fpath, kin_align = "none", dim_red_method = "none", alignment_method = "none", ndims= "none", **kwargs):
        # DONE: decoder should instantiate the model
        #DONE: instantiate the model like this :
        model = cfg.model_type(cfg.model_hyperparams) #TODO: pseudocode here replace with actual cfg implementation

        # kin_align_switch = {"refit": kinematic_alignment.ReFit,  # TODO: add back when stabilization is done
        #                     "none": kinematic_alignment.NoKinAlignment}

        # kin_align_method = kin_align_switch[kin_align]() # TODO: add back when stabilization is done
        
        # ls_extraction = latent_space_extraction.Custom(dim_red_method, alignment_method, ndims, **kwargs) # TODO: add back when stabilization is done

        # DONE by default decoder should load the model params
        self.model.load_model(fpath)

        super().__init__(model, None) # DONE: add stablization and dimensionality reduction when it is done

    def predict(self, neural_data):
        """
        Predict outputs given neural data in offline setting.

        Args:
            neural_data (numpy.ndarray): Input neural data of shape [N, numfeats]

        Returns:
            prediction (torch.Tensor): Predicted output
        """
        if(self.data_transformation is not None):
            transformed_data = torch.tensor(self.data_transformation.stabilize(neural_data)) #WARNING: Takes some time here to convert numpy to torch
        else:
            transformed_data = torch.tensor(neural_data)

        prediction = self.model.forward(transformed_data)
        
        return prediction

class StreamingDecoder(Decoder):
    def predict(self, neural_data):
        raise NotImplementedError
    
    def update(self, neural_data):
        raise NotImplementedError
    
class OnlineDecoder(Decoder):
    def predict(self, neural_data):
        raise NotImplementedError
  
    def decode_off(self):
        raise NotImplementedError
    
