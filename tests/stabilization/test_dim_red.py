from sklearn.decomposition import FactorAnalysis, PCA
import numpy as np
from neuraldecoding.stabilization.latent_space_alignment import dim_red

def test_factor_analysis():
    ## TODO: change ds to actual dataset
    ds = np.random([20, 96])
    
    ndims = 30
    lib = FactorAnalysis(n_components=ndims)
    lib_transformed = lib.fit_transform(ds)
    
    package = dim_red.FactorAnalysis(ndims)
    lm, _ = package.calc_lm(ds)
    package_transformed = package.reduce(ds, lm)
    
    assert np.allclose(lib_transformed == package_transformed)
    
def test_pca():
    ## TODO: change ds to actual dataset
    ds = np.random([20, 96])
    
    ndims = 30
    lib = PCA(n_components=ndims)
    lib_transformed = lib.fit_transform(ds)
    
    package = dim_red.PCA(ndims)
    lm, _ = package.calc_lm(ds)
    package_transformed = package.reduce(ds, lm)
    
    assert np.allclose(lib_transformed == package_transformed)
    
def test_proSVD():
    ## TODO
    raise NotImplementedError
   
    
def test_no_dim_red():
    ds = np.random([20, 96])

    package = dim_red.NoDimRed()
    lm = package.calc_lm(ds)
    package_transformed = package.reduce(ds, lm)
    
    assert np.allclose(ds == package_transformed)
    
def test_deg_fa():
    ## TODO
    raise NotImplementedError

def test_cca():
    ## TODO: change ds to actual dataset
    ds_X = np.random.rand(100, 50)
    ds_Y = np.random.rand(100, 60)
    
    package = dim_red.CanonicalCorrelationAnalysis()
    X_c, Y_c, cca = package.reduce(ds_X, ds_Y)
    
    assert X_c.shape[1] == Y_c.shape[1] == min(ds_X.shape[1], ds_Y.shape[1])
