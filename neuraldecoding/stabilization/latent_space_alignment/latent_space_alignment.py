from abc import ABC, abstractmethod
from ..stabilization import Stabilization

class LatentSpaceAlignment(Stabilization):
    def __init__(self, dim_red_method, alignment_method, ndims = None): 
        """ Initialize latent space alignment

        Inputs:
            dim_red_method ('DimRed' class): Dimensionality method
            alignment_method ('Alignment' class): Latent space alignemnt aethod
            ndims (int, optional): Number of dims to reduce to. Defaults to None.
        """        
        self.ndims = ndims
        self.dim_red_method = dim_red_method
        self.alignment_method = alignment_method
    
    def fit(self, data):
        """Train the LatentSpaceAlignment. Sets day_0 lm and returns latent space

        Args:
            data (numpy array of shape [n x timepoints]): day 0 neural data

        Returns:
            latent_ds (numpy array of shape [ndims x timepoints]): latent space
        """        
        if self.ndims is None:
            ndims = data.shape[0]
        else:
            ndims = self.ndims
            
        self.dim_red_method.set_dims(ndims)
        self.alignment_method.set_dims(ndims)

        lm, args = self.dim_red_method.calc_lm(data)
        self.alignment_method.set_baseline(lm)
        latent_ds = self.dim_red_method.reduce(data, lm, args)

        return latent_ds
    
    def extract_latent_space(self, data):
        """Extract and align a latent space

        Args:
            data (numpy array of shape [ndims x timepoints]): day k neural data

        Returns:
            ls (numpy array of shape [ndims x timepoints]): aligned latent space
        """        
        lm, args = self.dim_red_method.calc_lm(data)

        aligned_lm = self.alignment_method.get_aligned_lm(lm)

        latent_ds = self.dim_red_method.reduce(data, aligned_lm, args)
        
        return latent_ds
    
    def stabilize(self, data):
        data = self.extract_latent_space(data)
        return data
    
class PAF(LatentSpaceAlignment):
    def fit(self, data):
        """Train the LatentSpaceAlignment. Sets day_0 lm and returns latent space

        Args:
            data (numpy array of shape [n x timepoints]): day 0 data

        Returns:
            latent_ds (numpy array of shape [ndims x timepoints]): latent space
        """        
        if self.ndims is None:
            ndims = data.shape[0]
        else:
            ndims = self.ndims
            
        self.dim_red_method.setup(ndims)
        self.alignment_method.setup(ndims)

        lm, d, psi = self.dim_red_method.calc_lm(data)
        self.alignment_method.set_baseline(lm)
        latent_ds = self.dim_red_method.reduce(data, lm, d, psi)

        return latent_ds
    
    def extract_latent_space(self, data):
        """Extract and align a latent space

        Args:
            data (numpy array of shape [ndims x timepoints]): day k data

        Returns:
            ls (numpy array of shape [ndims x timepoints]): aligned latent space
        """        
        lm, d, psi = self.dim_red_method.calc_lm(data)
        
        aligned_lm = self.alignment_method.get_aligned_lm(lm)

        latent_ds = self.dim_red_method.reduce(data, aligned_lm, d, psi)
        
        return latent_ds
    
        