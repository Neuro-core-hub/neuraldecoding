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

##################
# Legacy Classes #
##################

class Decoder(ABC):
    def __init__(self, model, stabilization):
        """
        Decoder Class

        Args:
            model: model defined in neuraldecoding/model/
            stabilization: stabilization method #TODO
        """
        self.model = model
        self.stabilization = stabilization

    def load_decoder(self, fpath): #TODO how is path defined? like the one in adaptive alignment or ?
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
    def __init__(self, model_type, model_params, kin_align = "none", dim_red_method = "none", alignment_method = "none", ndims= "none", **kwargs):
        model_switch = {"KF": KalmanFilter(model_params),
                        #"SVM": models.SteadyStateKalmanFilter(), #TODO uncomment when it is implemented 
                        #"CNN": models.PybmiKalmanFilter(), #TODO uncomment when it is implemented 
                        "LSTM": LSTM(model_params)}

        # kin_align_switch = {"refit": kinematic_alignment.ReFit,  # TODO: add back when stabilization is done
        #                     "none": kinematic_alignment.NoKinAlignment}

        # kin_align_method = kin_align_switch[kin_align]() # TODO: add back when stabilization is done
        
        # ls_extraction = latent_space_extraction.Custom(dim_red_method, alignment_method, ndims, **kwargs) # TODO: add back when stabilization is done

        model = model_switch[model_type] 
        
        super().__init__(model, None)

    def predict(self, neural_data):
        """
        Predict outputs given neural data in offline setting.

        Args:
            neural_data (numpy.ndarray): Input neural data of shape [N, numfeats]

        Returns:
            prediction (torch.Tensor): Predicted output
        """
        if(self.stabilization is not None):
            stabilized_data = torch.tensor(self.stabilization.stabilize(neural_data)) #WARNING: Takes some time here to convert numpy to torch
        else:
            stabilized_data = torch.tensor(neural_data)

        prediction = self.model.forward(stabilized_data)
        
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
    
