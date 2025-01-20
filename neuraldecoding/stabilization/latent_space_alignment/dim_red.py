from abc import ABC, abstractmethod
import math
import numpy as np
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.decomposition import PCA as PrincipalComponentAnalyisis

# import adaptive_latents.prosvd as psvd

class DimRed(ABC):
    def __init__(self, ndims):
        """set ndims
        Args:
            ndims (int): number of dimensions to reduce to 
        """        
        self.ndims = ndims
        
class LoadingMatrixDimRed(DimRed):
    def apply(self, data):
        lm = self.calc_lm(data)
        reduced_data = self.reduce(data, lm)
        return reduced_data
    
class FactorAnalysis(LoadingMatrixDimRed):
    def calc_lm(self, data):
        """calulate loading matrix

        Args:
            data (numpy): neural data

        Returns:
            _type_: loading matrix which transforms
        """
        fa = FA(n_components=self.ndims)

        fa.fit(data)

        lm_partial = fa.components_
        psi = fa.noise_variance_
        W = lm_partial/psi
        loading_matrix = W.T @ np.linalg.inv(np.eye(len(lm_partial))+np.dot(W, lm_partial.T))

        return loading_matrix
    
    def reduce(self, data, lm):
        data_means = np.mean(data.neural, axis = 0)
        reduced_data = (data.neural - data_means) @ lm
        return reduced_data
    
class PCA(LoadingMatrixDimRed):
    def calc_lm(self, ds):
        pca = PrincipalComponentAnalyisis(n_components=self.ndims)
        pca.fit(ds.neural)
        lm = pca.components_.T
        
        return lm
    
    def reduce(self, data, lm):
        data_means = np.mean(data.neural, axis = 0)
        ls = (data.neural - data_means) @ lm
        ls_ds = data.make_latent_ds(ls)
        return ls_ds
    
class NoDimRed(LoadingMatrixDimRed):
    def calc_lm(self, data):
        return np.eye(data.neural.shape[1])
    
    def reduce(self, data, lm):
        return data
 
# TODO
# class ProSVD(DimRed):   
#     ## Note: this is very much setup for offline analysis, for online it will need to be altered
#     ## I have an idea of how to do this, but wanted to keep the initial implementation simple
#     def __init__(self, ndims, l = 1, l1 = .2):
#         """initialize ProSVD

#         Args:
#             l (int, optional): step size
#             l1 (float, optional): proportion of data to initialize on
#         """        
#         self.l = l
#         self.l1 = l1
#         super().__init__(ndims)

#     def get_lm(self, data):
#         pro = psvd.BaseProSVD(self.ndims)

#         n_init = int(data.neural.shape[0]*self.l1)
#         A_init = data.neural[:n_init, :].T
#         pro.initialize(A_init)

#         if self.l == -1:
#             l = data.neural.shape[0] - n_init
#         else:
#             l = self.l

#         if l == 0:
#             num_updates = 0

#         else:
#             num_updates = math.ceil((data.neural.shape[0] - n_init) / l)
            
#         for i in range(num_updates):
#             start_idx = (i * l)+n_init
#             end_idx = start_idx + l 
#             pro.updateSVD(data.neural[start_idx:end_idx, :].T)
 
#         return pro.Q
     
#     def reduce(self, data, lm):

#         ls = data @ lm

#         return ls
    

  
