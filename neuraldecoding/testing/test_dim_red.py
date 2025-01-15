from sklearn.decomposition import FactorAnalysis, PCA
import numpy as np
from ..stabilization.latent_space_alignment import dim_red

def test_factor_analysis():
    ## TODO: change ds to actual dataset
    ds = np.random([20, 96])
    
    ndims = 30
    lib = FactorAnalysis(n_components=ndims)
    lib_transformed = lib.fit_transform(ds)
    
    package = dim_red.FactorAnalysis()
    package.setup(ndims)
    lm = package.get_lm(ds)
    package_transformed = package.reduce(ds, lm)
    
    assert lib_transformed == package_transformed
    
def test_pca():
    ## TODO: change ds to actual dataset
    ds = np.random([20, 96])
    
    ndims = 30
    lib = PCA(n_components=ndims)
    lib_transformed = lib.fit_transform(ds)
    
    package = dim_red.PCA()
    package.setup(ndims)
    lm = package.get_lm(ds)
    package_transformed = package.reduce(ds, lm)
    
    assert lib_transformed == package_transformed
    
def test_proSVD():
    ## TODO
    raise NotImplementedError
   
    
def test_no_dim_red():
    ds = np.random([20, 96])
    
    ndims = 30
    lib = PCA(n_components=ndims)
    lib_transformed = lib.fit_transform(ds)
    
    package = dim_red.NoDimRed()
    package.setup(ndims)
    lm = package.get_lm(ds)
    package_transformed = package.reduce(ds, lm)
    
    assert ds == package_transformed    
    
    