from abc import abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import jit

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
    
class AdamGrad(Alignment):
    """
    Use adam gradient descent to align lm
    """   
    
    def __init__(self, init_method = 'procrustes', num_epochs = 500):
        self.init_method = init_method
        self.epochs = num_epochs
        self.name = 'adam_grad'

    def align(self, lm, data, path):
        epochs = self.epochs

        S = self.init_S(lm, data, path)
        Ss = []
        dists = np.empty([epochs+1])
        
        opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
        opt_state = opt_init(S)

        jit_loss = jit(self.loss)
        Ss.append(S)
        dists[0] = jit_loss(S, self.baseline, lm)

        for epoch in range(epochs):
            S = get_params(opt_state)  # Get current S from optimizer state
            deltaS = jax.grad(jit_loss)(S, self.baseline, lm)  # Compute gradient
            
            # Update optimizer state using the computed gradient
            opt_state = opt_update(epoch, deltaS, opt_state)
 
            Ss.append(S)
            dists[epoch+1] = jit_loss(S, self.baseline, lm)

        min_ind = np.argmin(dists)
        S = Ss[min_ind]
        
        aligned_lm = lm @ S.T
        
        self.dists = dists

        return aligned_lm
    
    @staticmethod
    def loss(S, baseline, day_k):
        return jnp.linalg.norm(baseline - day_k@S.T, ord = 'fro')
    
    def init_S(self, lm, data, path):
        if self.init_method == 'procrustes':
            m = self.baseline.T @ lm
            U, _, V = np.linalg.svd(m)
            
            S =  U @ V
        elif self.init_method == 'random':
            S = np.random.rand(self.baseline.shape[0], self.baseline.shape[0])
        elif self.init_method == 'identity':
            S = np.eye(self.baseline.shape[0])
        else:
            raise ValueError('Invalid init_method')
        
        return S
    
class NoAlignment(Alignment):
    """
    Don't perform any alignment
    """        
    
    def __init__(self):
        self.name = 'no_alignment'  
          
    def get_aligned_lm(self, lm):
        return lm 
    
    def align(self, lm):
        return lm
    
    
