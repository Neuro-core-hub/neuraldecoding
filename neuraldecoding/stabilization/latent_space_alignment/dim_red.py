from abc import ABC, abstractmethod
import math
import numpy as np
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.decomposition import PCA as PrincipalComponentAnalyisis
# sys.path.append('AdaptiveLatents')
# import adaptive_latents.prosvd as psvd

import warnings
# sys.path.append('fa_stable_manifolds_python')
from . import factor_analysis as fa_stable

class DimRed(ABC):
    def __init__(self, ndims = None):
        """set ndims
        Args:
            ndims (int): number of dimensions to reduce to 
        """        
        self.ndims = ndims

    def set_dims(self, ndims):
        self.ndims = ndims
        
class LoadingMatrixDimRed(DimRed):
    def apply(self, data):
        lm, args = self.calc_lm(data)
        reduced_data = self.reduce(data, lm, args)
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
        
        # warnings.warn("Warning: Debugging outputs, shouldn't occur in use, remove later")
        # communalities = np.sum(lm_partial**2, axis=0)

        # total_variance_captured = np.sum(communalities)
        # total_original_variance = np.sum(np.var(data, axis=0))

        # proportion_variance = total_variance_captured / total_original_variance
        # print(f"Total variance captured: {proportion_variance}")
        return loading_matrix, None
    
    def reduce(self, data, lm, args = None):
        if args is not None: 
            warnings.warn("Warning: Args passed to FactorAnalysis.reduce() when they are not needed")
        data_means = np.mean(data, axis = 0)
        reduced_data = (data - data_means) @ lm
        return reduced_data
    
class PCA(LoadingMatrixDimRed):
    def calc_lm(self, ds):
        print(ds.shape)
        pca = PrincipalComponentAnalyisis(n_components=self.ndims)
        pca.fit(ds)
        lm = pca.components_.T

        # warnings.warn("Warning: Debugging outputs, shouldn't occur in use, remove later")
        # cumvar = np.cumsum(pca.explained_variance_ratio_)
        # print(f"Cum Var of PCA: {cumvar}")
        print(lm.shape)
        return lm, None
    
    def reduce(self, data, lm, args = None):
        if args is not None: 
            warnings.warn("Warning: Args passed to PCA.reduce() when they are not needed")
        data_means = np.mean(data, axis = 0)
        ls = (data - data_means) @ lm
        return ls
    
class NoDimRed(LoadingMatrixDimRed):
    def calc_lm(self, data):
        return np.eye(data.shape[1]), None
    
    def reduce(self, data, lm, args):
        return data
 
# class ProSVD(DimRed):   
#     ## Note: this is very much setup for offline analysis, for online it will need to be altered
#     ## I have an idea of how to do this, but wanted to keep the initial implementation simple
#     def __init__(self, l = 1, l1 = .2):
#         """initialize ProSVD

#         Args:
#             l (int, optional): step size
#             l1 (float, optional): proportion of data to initialize on
#         """        
#         self.l = l
#         self.l1 = l1

#     def get_lm(self, data):
#         pro = psvd.BaseProSVD(self.ndims)

#         n_init = int(data.shape[0]*self.l1)
#         A_init = data[:n_init, :].T
#         pro.initialize(A_init)

#         if self.l == -1:
#             l = data.shape[0] - n_init
#         else:
#             l = self.l

#         if l == 0:
#             num_updates = 0

#         else:
#             num_updates = math.ceil((data.shape[0] - n_init) / l)
            
#         for i in range(num_updates):
#             start_idx = (i * l)+n_init
#             end_idx = start_idx + l 
#             pro.updateSVD(data[start_idx:end_idx, :].T)
 
#         return pro.Q
     
#     def reduce(self, data, lm):

#         ls = data @ lm

#         return ls
    
  
class EMFactorAnalysis1(DimRed):
    """
    This is the factor analysis version used by procrustes alignment of factors (Degenhart 2022)
    """

    def __init__(self, n_restarts = 5, max_n_inits = 300, ll_diff_threshold = .01, min_priv_var = .1, verbose = False):
        self.n_restarts = n_restarts
        self.max_n_inits = max_n_inits
        self.ll_diff_threshold = ll_diff_threshold
        self.min_priv_var = min_priv_var
        self.verbose = verbose
        
    def calc_lm(self, data):
        d, base_lm, psi, _, _ = fa_stable.get_factor_analysis_loading(data, self.ndims, max_n_its = self.max_n_inits, 
                                        ll_diff_thresh=self.ll_diff_threshold, min_priv_var=self.min_priv_var, verbose = self.verbose)
        print(base_lm.shape)
        print(base_lm)
        return base_lm, (d, psi)
    
    def reduce(self, data, lm, args):
        d, psi = args
        regularized_lm, O = fa_stable.get_stabilization_matrices(lm, psi, d)
        reduced_data = data @ regularized_lm.T + np.expand_dims(O, axis = -1).T
        return reduced_data
    