from abc import abstractmethod

import numpy as np

class Alignment():
    def __init__():
        raise NotImplementedError
    
    def setup(self, ndims):
        """set ndims

        Args:
            ndims (int): number of dimensions to reduce to 
        """        
        self.ndims = ndims
        
    def set_baseline(self, lm):
        """_Set day 0 loading matrix

        Args:
            lm (Numpy array of shape []): day_0 loading matrix
        """        
        self.baseline = lm
        
    def get_aligned_lm(self, lm):
        a_lm = self.align(lm)
        return a_lm

    @abstractmethod
    def align(self, lm):
        raise NotImplementedError

class ProcrustesAlignment(Alignment):
    """
    Use orthogonal procrustes to align day_0 lm to day_k lm
    """    
    def align(self, lm):              
        m = self.baseline @ lm.T
        U, _, V = np.linalg.svd(m)
        
        S = V.T @ U.T
        
        aligned_lm = S.T @ lm
        
        return aligned_lm
    
class SwitchLM(Alignment):
    """
    Use day_0 lm
    """   
    def __init__(self):
        self.name = "switch"

    def align(self, lm):
        return self.baseline
    
class NoAlignment(Alignment):
    """
    Don't perform any alignment
    """            
    def get_aligned_lm(self, lm):
        return lm 
    
    def align(self, lm):
        return lm
    
    
