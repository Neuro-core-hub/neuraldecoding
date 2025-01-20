from abc import ABC, abstractmethod

class Stabilization(ABC):
    def __init__():
        raise NotImplementedError
    
    @abstractmethod
    def stabilize(input):
        raise NotImplementedError
    

class LatentSpaceALignment(Stabilization):
    def __init__(self, dim_red_method, alignment_method):
        self.dim_red_method = dim_red_method
        self.alignment_method = alignment_method
        
    def train(self, data):
        """Train latent space alignment

        Args:
            data (np array): day 0 neural data
        """
        lm = self.dim_red_method.get_lm(data)
        self.alignment_method.set_baseline(lm)
        
    def stabilize(self, data):
        """Perfrorm latent space alignment

        Args:
            data (np array): day k neural data to be aligned to day 0

        Returns:
            np array: aligned neural data
        """
        lm = self.dim_red_method.get_lm(data)
        aligned_lm = self.alignment_method.align(lm)
        aligned_data = self.dim_red_method.reduce(data, aligned_lm) 
        
        return aligned_data

        