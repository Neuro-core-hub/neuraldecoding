import unittest
from unittest.mock import patch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from neuraldecoding.preprocessing.blocks import NormalizationBlock
import tempfile
import pickle

class TestNormalizationBlock(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, 'normalizer.pkl')
        
    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        os.rmdir(self.temp_dir)
        
    def test_sklearn_normalization_train(self):
        location = ['train', 'test']
        normalizer_params = {
            'type': 'StandardScaler',
            'params': {},
            'is_save': True,
            'save_path': self.save_path
        }
        block = NormalizationBlock(location, 'sklearn', normalizer_params)
        
        data = {
            'train': np.random.randn(100, 10),
            'test': np.random.randn(50, 10)
        }
        interpipe = {'is_train': True}
        
        result_data, result_interpipe = block.transform(data, interpipe)
        _, _ = block.transform(data, interpipe)
        
        # Verify the data was processed
        self.assertEqual(result_data['train'].shape, data['train'].shape)
        self.assertEqual(result_data['test'].shape, data['test'].shape)
            
    def test_sequence_scaler_normalization_train(self):
        location = ['train', 'test']
        normalizer_params = {
            'is_save': True,
            'save_path': self.save_path
        }
        block = NormalizationBlock(location, 'sequence_scaler', normalizer_params)
        
        data = {
            'train': np.random.randn(100, 10, 20),
            'test': np.random.randn(50, 10, 20)
        }
        interpipe = {'is_train': True}
        
        result_data, result_interpipe = block.transform(data, interpipe)
        
        # Verify the data was processed
        self.assertEqual(result_data['train'].shape, data['train'].shape)
        self.assertEqual(result_data['test'].shape, data['test'].shape)
            
    def test_unsupported_method_raises_error(self):
        location = ['train', 'test']
        normalizer_params = {'is_save': False}
        block = NormalizationBlock(location, 'unsupported_method', normalizer_params)
        
        data = {'train': np.random.randn(10, 5), 'test': np.random.randn(5, 5)}
        interpipe = {'is_train': True}
        
        with self.assertRaises(ValueError):
            block.transform(data, interpipe)
            
    def test_missing_save_path_raises_error(self):
        location = ['train', 'test']
        normalizer_params = {'type': 'StandardScaler',
                                'params': {},
                                'is_save': True}
        block = NormalizationBlock(location, 'sklearn', normalizer_params)
        
        data = {'train': np.random.randn(10, 5), 'test': np.random.randn(5, 5)}
        interpipe = {'is_train': True}
        
        with self.assertRaises(ValueError):
            block.transform(data, interpipe)

if __name__ == "__main__":
    unittest.main()
