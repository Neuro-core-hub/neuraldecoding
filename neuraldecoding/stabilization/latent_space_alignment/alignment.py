from abc import abstractmethod

import numpy as np

class Alignment():
    def __init__():
        raise NotImplementedError
    
    def set_dims(self, ndims):
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
    def __init__(self):
        self.name = "procrustes_alignment"

    def align(self, lm):              
        m = self.baseline.T @ lm
        U, _, V = np.linalg.svd(m)
        
        S =  U @ V
        
        aligned_lm = lm @ S.T
        
        return aligned_lm
    
class SwitchLM(Alignment):
    """
    Use day_0 lm
    """   
    def __init__(self):
        self.name = "switch_alignment"

    def align(self, lm):
        return self.baseline
    
class NoAlignment(Alignment):
    """
    Don't perform any alignment
    """          
    def __init__(self):
        self.name = "no_alignment"  

    def align(self, lm):
        return lm
    
    
