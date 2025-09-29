from abc import ABC, abstractmethod
import neuraldecoding.stabilization.latent_space_alignment.dim_red
import neuraldecoding.stabilization.latent_space_alignment.alignment
from neuraldecoding.stabilization import Stabilization
import pickle
import time

class LatentSpaceAlignment(Stabilization):
    def __init__(self, cfg): 
        """ Initialize latent space alignment

        Inputs:
            dim_red_method ('DimRed' class): Dimensionality method
            alignment_method ('Alignment' class): Latent space alignemnt aethod
            ndims (int, optional): Number of dims to reduce to. Defaults to None.
        """        
        self.ndims = cfg.ndims
        self.dim_red_method = getattr(neuraldecoding.stabilization.latent_space_alignment.dim_red, cfg.dim_red_method.type)(**cfg.dim_red_method.params)
        self.alignment_method = getattr(neuraldecoding.stabilization.latent_space_alignment.alignment, cfg.alignment_method.type)(**cfg.alignment_method.params)
        self.fpath = cfg.fpath

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
        print(f"aabaseline to be set: {lm}")
        self.alignment_method.set_baseline(lm)
        latent_ds = self.dim_red_method.reduce(data, lm, args)
        print(f"Latent space shape: {latent_ds.shape}")
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
        print(f"baslm and aligned lm shapes {lm.shape} and {aligned_lm.shape}")

        latent_ds = self.dim_red_method.reduce(data, aligned_lm, args)

        return latent_ds
    
    def stabilize(self, data):
        data = self.extract_latent_space(data)
        return data
    
    def save_alignment(self):
        with open(self.fpath, 'wb') as f:
            pickle.dump({
                'dim_red_method': self.dim_red_method,
                'alignment_method': self.alignment_method,
                'ndims': self.ndims
            }, f)

    def load_alignment(self):
        with open(self.fpath, 'rb') as f:
            data = pickle.load(f)
            self.dim_red_method = data['dim_red_method']
            self.alignment_method = data['alignment_method']
            self.ndims = data['ndims']
    
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
    
        