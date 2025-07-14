import unittest
from sklearn.decomposition import FactorAnalysis, PCA
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from neuraldecoding.stabilization.latent_space_alignment import dim_red

class TestDimRed(unittest.TestCase):
    
    def test_factor_analysis(self):
        ## TODO: change ds to actual dataset
        ds = np.random.random([20, 96])
        
        ndims = 30
        lib = FactorAnalysis(n_components=ndims)
        lib_transformed = lib.fit_transform(ds)
        
        package = dim_red.FactorAnalysis(ndims)
        lm, _ = package.calc_lm(ds)
        package_transformed = package.reduce(ds, lm)
        
        self.assertTrue(np.allclose(lib_transformed, package_transformed))
        
    def test_pca(self):
        ## TODO: change ds to actual dataset
        ds = np.random.random((100, 96))

        ndims = 30
        lib = PCA(n_components=ndims)
        lib_transformed = lib.fit_transform(ds)
        
        package = dim_red.PCA(ndims)
        lm, _ = package.calc_lm(ds)
        package_transformed = package.reduce(ds, lm)
        
        self.assertTrue(np.allclose(lib_transformed, package_transformed))
       
    def test_no_dim_red(self):
        ds = np.random.random([20, 96])

        package = dim_red.NoDimRed()
        lm = package.calc_lm(ds)
        package_transformed = package.reduce(ds, lm, args={})
        
        self.assertTrue(np.allclose(ds, package_transformed))
        
    def test_deg_fa(self):
        ds = np.linspace(0, 1, 100*96).reshape(100, 96)
        ndims = 30
        package = dim_red.EMFactorAnalysis1()
        package.set_dims(ndims)
        lm, args = package.calc_lm(ds)
        package_transformed = package.reduce(ds, lm, args)
        
        self.assertEqual(package_transformed.shape, (100, 30))

if __name__ == '__main__':
    unittest.main()