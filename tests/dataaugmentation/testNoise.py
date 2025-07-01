import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from neuraldecoding.dataaugmentation import DataAugmentation as da
from tests.dataaugmentation.UtilsTestDataAugmentation import generateData

import numpy as np
import unittest


class TestNoise(unittest.TestCase):
    """
    Unit test class for verifying different numpy add noise functions in DataAugmentation.

    This test suite covers:
        - add_noise_white
        - add_noise_random_walk
        - add_noise_constant (both 'same' and 'different' bias types)
    """
    def setUp(self):
        """
        Set up function that sets seed and generates synthetic test data for all test cases.

        The data is simulated based on distributions extracted from `2024-02-05_preprocess.pkl`:
        - self.x: (22321, 96), Gaussian with mean = 11.1161 and std = 15.6922 (sbp)
        - self.y: (22321, 4), Gaussian with mean = 0.2433 and std = 0.2706 (finger kinematics)
        """
        self.seed = 12345
        np.random.seed(self.seed)

        nTimeBins = 22321
        mu_x = 11.116120635071743
        std_x = 15.69221856007768
        mu_y = 0.2433446115719394
        std_y = 0.27057285460646824

        self.x = generateData(nTimeBins,96,'gaussian',mu_x,std_x)
        self.y = generateData(nTimeBins,4,'gaussian',mu_y,std_y)

        
    def test_noise_white(self):
        """
        Tests add_noise_white by verifying:
            - Determinism given fixed seed
            - Zero mean of noise
            - Expected standard deviation of noise
            - Output shape matches input
            - Consistent behavior across a range of std values
        """
        X = self.x
        std = 4

        noise_X = da.add_noise_white(X, std)

        # Test 1: determinism
        np.random.seed(self.seed)
        result_1 = da.add_noise_white(X,std)
        np.random.seed(self.seed)
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
        """
        Tests add_noise_random_walk by verifying:
            - Determinism given fixed seed
            - Noise is a smooth random walk with ~zero mean of diff
            - Observed std of diff noise matches expected
            - Output shape matches input
            - Consistent behavior across multiple std values
        """
        std = 1
        X = self.x
        noise_X = da.add_noise_random_walk(X, std)

        # Test 1: determinism
        np.random.seed(self.seed)
        result_1 = da.add_noise_random_walk(X,std)
        np.random.seed(self.seed)
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
        """
        Tests add_noise_constant by verifying:
            - Determinism given fixed seed for both 'same' and 'different' bias types
            - Output shape matches input for both types
            - For 'same': constant offset per channel over time
            - For 'different': different offsets across channels
            - Invalid bias type defaults to 'same'
            - Mean and std of added noise match expectations across multiple stds
        """
        std = 1
        X = self.x

        # Test 1: determinism
        np.random.seed(self.seed)
        result_1 = da.add_noise_constant(X,std,'same')
        np.random.seed(self.seed)
        result_2 = da.add_noise_constant(X,std,'same')
        np.testing.assert_array_equal(result_1, result_2, "test add_noise_constant (same): not following determinism")

        np.random.seed(self.seed)
        result_1 = da.add_noise_constant(X,std,'different')
        np.random.seed(self.seed)
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

        # Test 6: Mean and Std
        std_values = [0.01, 0.1, 1.0, 2.0, 4.0]
        for bias_type in ['same', 'different']:
            for std in std_values:
                np.random.seed(self.seed)
                bias_X = da.add_noise_constant(X, std, type=bias_type)
                noise = bias_X - X
                if bias_type == 'different':
                    noise_mean = np.mean(noise)
                    assert abs(noise_mean) < 0.2, f"test add_noise_constant: Mean not ~0 for 'different', got {noise_mean}"
                    noise_std = np.std(noise)
                    assert np.allclose(noise_std, std, rtol=0.2), \
                        f"test add_noise_constant: Std mismatch for 'different', expected={std}, got {noise_std}"
                elif bias_type == 'same':
                    noise_std = np.std(noise)
                    assert noise_std < 1e-6, \
                        f"test add_noise_constant: Std not ~0 for 'same', got {noise_std}"

if __name__ == '__main__':
    unittest.main()
