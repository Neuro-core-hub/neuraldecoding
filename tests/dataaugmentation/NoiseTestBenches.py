import torch
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from neuraldecoding.dataaugmentation import DataAugmentation as da
from matplotlib import pyplot as plt
import TestDataAugmentation as tda
import unittest
import unittest
import numpy as np
from TestDataAugmentation import generateData, plotData
from sklearn.preprocessing import MinMaxScaler
import os
import pickle



class TestDA(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345)

        with open(os.path.join('C:\\Files\\UM\\ND\\DataAugmentation\\Data', '2024-02-05_preprocess.pkl'), 'rb') as f:
            data_CO, data_RD = pickle.load(f)
        self.x = data_CO['sbp']
        self.y = data_CO['finger_kinematics']
        
    def test_noise_white(self):
        
        std = 4
        X = self.x
        noise_X = da.add_noise_white(X, std)

        # Test 1: determinism
        np.random.seed(12345)
        result_1 = da.add_noise_white(X,std)
        np.random.seed(12345)
        result_2 = da.add_noise_white(X,std)
        np.testing.assert_array_equal(result_1, result_2, "test add_noise_white: not following determinism")

        # Test 2: mean
        assert abs(np.mean(noise_X - X) - 0) < 0.1 , "test add_noise_white: mean should be zero"
        
        # Test 3: std
        assert abs(np.std(noise_X - X) - std) < 0.1 , "test add_noise_white: std should match"

        # Test 4: shape
        assert noise_X.shape == X.shape , "test add_noise_white: shape not match"

        # Test 5: different std values
        std_values = [0.01, 0.1, 1.0, 2.0, 4.0]
        for std in std_values:
            noise_X = da.add_noise_white(X, std)
            noise = noise_X - X
            observed_std = np.std(noise)
            assert np.allclose(observed_std, std, rtol=0.2), f"Failed for std={std}"

    def test_noise_random_walk(self):
        
        std = 1
        X = self.x
        noise_X = da.add_noise_random_walk(X, std)

        # Test 1: determinism
        np.random.seed(12345)
        result_1 = da.add_noise_random_walk(X,std)
        np.random.seed(12345)
        result_2 = da.add_noise_random_walk(X,std)
        np.testing.assert_array_equal(result_1, result_2, "test add_noise_random_walk: not following determinism")
        
        # Test 2: mean
        noise = noise_X - X
        noise_diff = np.diff(noise, axis=0)
        assert np.allclose(np.mean(noise_diff), 0, atol=0.1), "test add_noise_random_walk:Noise differences should have mean close to 0"
        
        # Test 3: std
        observed_std = np.std(noise_diff)
        assert np.allclose(observed_std, std, rtol=0.2), f"test add_noise_random_walk:Expected std {std}, got {observed_std}"
        
        # Test 4: shape
        assert noise_X.shape == X.shape, f"test add_noise_random_walk:Expected shape {X.shape}, got {noise_X.shape}"

        # Test 5: different std values
        std_values = [0.01, 0.1, 1.0, 2.0, 4.0]
        for std in std_values:
            noise_X = da.add_noise_random_walk(X, std)
            noise = noise_X - X
            noise_diff = np.diff(noise, axis=0)
            observed_std = np.std(noise_diff)
            assert np.allclose(observed_std, std, rtol=0.2), f"test add_noise_random_walk:Failed for std={std}"

    def test_noise_constant(self):
        std = 1
        X = self.x

        # Test 1: determinism
        np.random.seed(12345)
        result_1 = da.add_noise_constant(X,std,'same')
        np.random.seed(12345)
        result_2 = da.add_noise_constant(X,std,'same')
        np.testing.assert_array_equal(result_1, result_2, "test add_noise_constant (same): not following determinism")

        np.random.seed(12345)
        result_1 = da.add_noise_constant(X,std,'different')
        np.random.seed(12345)
        result_2 = da.add_noise_constant(X,std,'different')
        np.testing.assert_array_equal(result_1, result_2, "test add_noise_constant (different): not following determinism")
        
        # Test 2: output shape for both types
        for bias_type in ['same', 'different']:
            bias_X = da.add_noise_constant(X, std, type=bias_type)
            assert bias_X.shape == X.shape, f"test add_noise_constant:shape mismatch for type={bias_type}"
        
        # Test 3: verify 'same'
        bias_X = da.add_noise_constant(X, std, type='same')
        noise = bias_X - X
        
        for channel in range(X.shape[1]):
            channel_noise = noise[:, channel]
            assert np.allclose(channel_noise, channel_noise[0]), f"test add_noise_constant:Channel {channel} noise not constant across time for 'same' type"
        
        # Test 4: verify 'different'
        bias_X = da.add_noise_constant(X, std, type='different')
        noise = bias_X - X
        
        channel_means = np.mean(noise, axis=0)
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                assert not np.allclose(channel_means[i], channel_means[j]), f"test add_noise_constant:Channels {i} and {j} have same noise for 'different' type"
        
        # Test 5: Test invalid type handling
        bias_X = da.add_noise_constant(X, std, type='invalid')
        noise = bias_X - X
        # Should default to 'same' behavior
        for channel in range(X.shape[1]):
            channel_noise = noise[:, channel]
            assert np.allclose(channel_noise, channel_noise[0]), "test add_noise_constant:Invalid type didn't default to 'same' behavior"


if __name__ == '__main__':
    unittest.main()
