from abc import ABC, abstractmethod
import math
import numpy as np
import adaptive_latents.prosvd as psvd

class DimRed(ABC):
    def __init__():
        pass
    
    def setup(self, ndims):
        """set ndims

        Args:
            ndims (int): number of dimensions to reduce to 
        """        
        self.ndims = ndims
        
    @abstractmethod
    def get_lm(self, data):
        """Get loadimg matrix

        Args:
            data (numpy array of shape [n x t]): Full dimensionality data

        Returns:
            lm (numpy array of shape []): Latent space loading matrix
        """        
        raise NotImplementedError
        return lm
    
    @abstractmethod
    def reduce(self, data, lm):
        """Perform dimensionality reduction

        Args:
            data (numpy array of shape [n x t]): Full dimensionality data 
            lm (numpy array of shape []): Latent space loading matrix

        Returns:
            ls (numpy array of shape [ndims x timepoints]): latent space
        """        
        raise NotImplementedError
        return ls
    
   
class ProSVD(DimRed):   
    ## Note: this is very much setup for offline analysis, for online it will need to be altered
    ## I have an idea of how to do this, but wanted to keep the initial implementation simple
    def __init__(self, l = 1, l1 = .2):
        """initialize ProSVD

        Args:
            l (int, optional): step size
            l1 (float, optional): proportion of data to initialize on
        """        
        self.l = l
        self.l1 = l1

    def get_lm(self, data):
        pro = psvd.BaseProSVD(self.ndims)

        n_init = int(data.neural.shape[0]*self.l1)
        A_init = data.neural[:n_init, :].T
        pro.initialize(A_init)

        if self.l == -1:
            l = data.neural.shape[0] - n_init
        else:
            l = self.l

        if l == 0:
            num_updates = 0

        else:
            num_updates = math.ceil((data.neural.shape[0] - n_init) / l)
            
        for i in range(num_updates):
            start_idx = (i * l)+n_init
            end_idx = start_idx + l 
            pro.updateSVD(data.neural[start_idx:end_idx, :].T)
 
        return pro.Q
     
    def reduce(self, data, lm):

        ls = data @ lm

        return ls
    
class FactorAnalysis(DimRed):
    def get_lm(self, data):
        raise NotImplementedError
    
    def reduce(self, data, lm):
        raise NotImplementedError
    
class PCA(DimRed):
    def get_lm(self, data):
        raise NotImplementedError
    
    def reduce(self, data, lm):
        raise NotImplementedError
    
class NoDimRed(DimRed):
    def get_lm(self, data):
        raise NotImplementedError
    
    def reduce(self, data, lm):
        raise NotImplementedError
 