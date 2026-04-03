from abc import abstractmethod
import numpy as np
import pickle
import os
# import jax
# import jax.numpy as jnp
# from jax.example_libraries import optimizers
# from jax import jit

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
        print("Aligning with Procrustes") 
        print(f"baseline: {self.baseline}")
        m = self.baseline.T @ lm
        U, _, V = np.linalg.svd(m)
        
        S =  U @ V
        
        aligned_lm = lm @ S.T
        self.save_dict = {'U': U, 'V': V, 'S': S, 'aligned_lm': aligned_lm, 'lm': lm, 'baseline': self.baseline, 
                          'pre_aligned_norm_from_baseline':np.linalg.norm(lm - self.baseline, ord = 'fro'), 
                          'post_aligned_norm_from_baseline': np.linalg.norm(aligned_lm - self.baseline, ord='fro')}
        
        return aligned_lm

class TestProcrustesAlignment(Alignment):
    """
    Use orthogonal procrustes to align day_0 lm to day_k lm
    """  
    def __init__(self):
        self.name = "procrustes_alignment"  
        
    def align(self, lm):     
        m1 = self.baseline
        m2 = lm
        S = np.matmul(m1.T, m2)
        U, _, V = np.linalg.svd(S)
        V = V.T # due to differences in Matlab and numpy impls 
        T = np.matmul(U, V.T)
        aligned_lm = np.matmul(m2, T.T)

        return aligned_lm
    
class SwitchLM(Alignment):
    """
    Use day_0 lm
    """   
    def __init__(self):
        self.name = "switch_alignment"

    def align(self, lm):
        return self.baseline
    
# class AdamGrad(Alignment):
#     """
#     Use adam gradient descent to align lm
#     """   
    
#     def __init__(self, init_method = 'procrustes', num_epochs = 500):
#         self.init_method = init_method
#         self.epochs = num_epochs
#         self.name = 'adam_grad'

#     def align(self, lm, data, path):
#         epochs = self.epochs

#         S = self.init_S(lm, data, path)
#         Ss = []
#         dists = np.empty([epochs+1])
        
#         opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
#         opt_state = opt_init(S)

#         jit_loss = jit(self.loss)
#         Ss.append(S)
#         dists[0] = jit_loss(S, self.baseline, lm)

#         for epoch in range(epochs):
#             S = get_params(opt_state)  # Get current S from optimizer state
#             deltaS = jax.grad(jit_loss)(S, self.baseline, lm)  # Compute gradient
            
#             # Update optimizer state using the computed gradient
#             opt_state = opt_update(epoch, deltaS, opt_state)
 
#             Ss.append(S)
#             dists[epoch+1] = jit_loss(S, self.baseline, lm)

#         min_ind = np.argmin(dists)
#         S = Ss[min_ind]
        
#         aligned_lm = lm @ S.T
        
#         self.dists = dists

#         return aligned_lm
    
#     @staticmethod
#     def loss(S, baseline, day_k):
#         return jnp.linalg.norm(baseline - day_k@S.T, ord = 'fro')
    
#     def init_S(self, lm, data, path):
#         if self.init_method == 'procrustes':
#             m = self.baseline.T @ lm
#             U, _, V = np.linalg.svd(m)
            
#             S =  U @ V
#         elif self.init_method == 'random':
#             S = np.random.rand(self.baseline.shape[0], self.baseline.shape[0])
#         elif self.init_method == 'identity':
#             S = np.eye(self.baseline.shape[0])
#         else:
#             raise ValueError('Invalid init_method')
        
#         return S
    
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
    
    
class CCA(Alignment):
    def __init__(self, cfg = None):
        self.name = 'cca'
    
    def set_baseline(self, ds):
        self.baseline = ds
        self.Q0, self.R0 = np.linalg.qr(ds)

    def align(self, ls):
        assert ls.shape[0] > ls.shape[1], "Data should be in shape [n_timepoints x n_latent_dimensions]"

        Qk, Rk = np.linalg.qr(ls)

        ## TO delete
        shape = min(Qk.shape[0], self.Q0.shape[0])
        Q0 = self.Q0[:shape, :]
        Qk = Qk[:shape, :]
        ##
        
        U, _, Vt = np.linalg.svd(Q0.T @ Qk)
        Mk = np.linalg.inv(Rk) @ Vt.T
        M0 = np.linalg.inv(self.R0) @ U
        # base_ls = self.baseline @ M0
        aligned_ls = ls @ Mk @ np.linalg.inv(M0)
        self.save_dict = {
            'U': U,
            'Vt': Vt,
            'Mk': Mk,
            'M0': M0,
            'aligned_lm': aligned_ls,
            'lm': ls,
            'baseline': self.baseline,
            'pre_aligned_norm_from_baseline': np.linalg.norm(ls - self.baseline, ord='fro'),
            'post_aligned_norm_from_baseline': np.linalg.norm(aligned_ls - self.baseline, ord='fro')
        }
        
        return aligned_ls



class NeuralNetworkAlignment(Alignment):
    """
    Nonlinear latent alignment via a small neural network (MLP)
    Maps day_k latent space -> day_0 latent space
    """
    def __init__(self, hidden_sizes=[64, 64], lr=1e-3, epochs=100):
        self.name = "nn_alignment"
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def set_baseline(self, lm):
        """
        Baseline is day_0 latent space (shape: [channels x ndims])
        """
        self.baseline = lm
        self.ndims = lm.shape[1]

    def get_aligned_lm(self, lm):
        return self.align(lm)

    def align(self, lm):
        """
        lm: day_k latent space [channels x ndims]
        returns aligned latent space
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(lm, dtype=torch.float32, device=device)
        Y = torch.tensor(self.baseline, dtype=torch.float32, device=device)

        # Define simple MLP
        layers = []
        in_dim = self.ndims
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, self.ndims))  # output dim = ndims
        model = nn.Sequential(*layers).to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # Training
        for _ in range(self.epochs):
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)
            loss.backward()
            optimizer.step()

        # Return aligned latent space
        with torch.no_grad():
            aligned = model(X).cpu().numpy()
        self.save_dict = {
            'aligned_lm': aligned,
            'lm': lm,
            'baseline': self.baseline,
            'pre_aligned_norm_from_baseline': np.linalg.norm(lm - self.baseline, ord='fro'),
            'post_aligned_norm_from_baseline': np.linalg.norm(aligned - self.baseline, ord='fro')
        }
        return aligned
