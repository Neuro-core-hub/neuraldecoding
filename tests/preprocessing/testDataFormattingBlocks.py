import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
from neuraldecoding.preprocessing.blocks import (
    Dict2DataDictBlock, ClassificationDict2TupleBlock, DataKeyRenameBlock,
    InterpipeKeyRenameBlock, DataSplitBlock, Dict2TupleBlock, Dataset2DictBlock
)

class TestDict2DataDictBlock(unittest.TestCase):
    def setUp(self):
        self.block = Dict2DataDictBlock()
        self.mock_data = {
            'sbp': np.random.randn(100, 10),
            'behaviour': np.random.randn(100, 3),
            'trial_info': np.arange(100)
        }
        
    def test_transform(self):
        import neuraldecoding.utils
        original_func = getattr(neuraldecoding.utils, 'neural_finger_from_dict', None)
        
        def mock_neural_finger_from_dict(data, neural_type):
            return (data['sbp'], data['behaviour']), data['trial_info']
        
        neuraldecoding.utils.neural_finger_from_dict = mock_neural_finger_from_dict
        
        try:
            interpipe = {}
            result_data, result_interpipe = self.block.transform(self.mock_data, interpipe)
            
            self.assertIn('neural', result_data)
            self.assertIn('behaviour', result_data)
            self.assertIn('trial_idx', result_interpipe)
            np.testing.assert_array_equal(result_data['neural'], self.mock_data['sbp'])
            np.testing.assert_array_equal(result_data['behaviour'], self.mock_data['behaviour'])
            np.testing.assert_array_equal(result_interpipe['trial_idx'], self.mock_data['trial_info'])
        finally:
            if original_func:
                neuraldecoding.utils.neural_finger_from_dict = original_func

class TestDataKeyRenameBlock(unittest.TestCase):
    def setUp(self):
        self.rename_map = {'old_key1': 'new_key1', 'old_key2': 'new_key2'}
        self.block = DataKeyRenameBlock(self.rename_map)
        
    def test_transform(self):
        data = {'old_key1': 'value1', 'old_key2': 'value2', 'unchanged': 'value3'}
        interpipe = {}
        
        result_data, result_interpipe = self.block.transform(data, interpipe)
        
        self.assertIn('new_key1', result_data)
        self.assertIn('new_key2', result_data)
        self.assertIn('unchanged', result_data)
        self.assertNotIn('old_key1', result_data)
        self.assertNotIn('old_key2', result_data)
        self.assertEqual(result_data['new_key1'], 'value1')
        self.assertEqual(result_data['new_key2'], 'value2')
        
    def test_missing_key(self):
        data = {'some_other_key': 'value'}
        interpipe = {}
        
        with self.assertRaises(KeyError):
            self.block.transform(data, interpipe)

class TestInterpipeKeyRenameBlock(unittest.TestCase):
    def setUp(self):
        self.rename_map = {'old_key1': 'new_key1', 'old_key2': 'new_key2'}
        self.block = InterpipeKeyRenameBlock(self.rename_map)
        
    def test_transform(self):
        data = {'some_data': 'value'}
        interpipe = {'old_key1': 'value1', 'old_key2': 'value2', 'unchanged': 'value3'}
        
        result_data, result_interpipe = self.block.transform(data, interpipe)
        
        self.assertIn('new_key1', result_interpipe)
        self.assertIn('new_key2', result_interpipe)
        self.assertIn('unchanged', result_interpipe)
        self.assertNotIn('old_key1', result_interpipe)
        self.assertNotIn('old_key2', result_interpipe)
        self.assertEqual(result_interpipe['new_key1'], 'value1')
        self.assertEqual(result_interpipe['new_key2'], 'value2')
        
    def test_missing_key(self):
        data = {'some_data': 'value'}
        interpipe = {'some_other_key': 'value'}
        
        with self.assertRaises(KeyError):
            self.block.transform(data, interpipe)

class TestDataSplitBlock(unittest.TestCase):
    def setUp(self):
        self.block = DataSplitBlock(split_ratio=0.8, split_seed=42)
        
    def test_transform(self):
        neural_data = np.random.randn(100, 10)
        finger_data = np.random.randn(100, 3)
        trial_idx = np.arange(100)
        
        data = {'neural': neural_data, 'behaviour': finger_data}
        interpipe = {'trial_idx': trial_idx}
        
        import neuraldecoding.utils
        original_func = getattr(neuraldecoding.utils, 'data_split_trial', None)
        
        def mock_data_split_trial(neural, finger, trial_idx, split_ratio, seed):
            split_point = int(len(neural) * split_ratio)
            return ((neural[:split_point], finger[:split_point]), 
                   (neural[split_point:], finger[split_point:]))
        
        neuraldecoding.utils.data_split_trial = mock_data_split_trial
        
        try:
            result_data, result_interpipe = self.block.transform(data, interpipe)
            
            self.assertIn('neural_train', result_data)
            self.assertIn('neural_test', result_data)
            self.assertIn('behaviour_train', result_data)
            self.assertIn('behaviour_test', result_data)
            
            self.assertEqual(len(result_data['neural_train']), 80)
            self.assertEqual(len(result_data['neural_test']), 20)
            self.assertEqual(len(result_data['behaviour_train']), 80)
            self.assertEqual(len(result_data['behaviour_test']), 20)
        finally:
            if original_func:
                neuraldecoding.utils.data_split_trial = original_func
                
    def test_missing_trial_idx(self):
        data = {'neural': np.random.randn(100, 10), 'behaviour': np.random.randn(100, 3)}
        interpipe = {}
        
        with self.assertRaises(ValueError):
            self.block.transform(data, interpipe)

class TestDict2TupleBlock(unittest.TestCase):
    def setUp(self):
        self.block = Dict2TupleBlock()
        
    def test_transform_two_keys(self):
        data = {'neural': np.random.randn(100, 10), 'behaviour': np.random.randn(100, 3)}
        interpipe = {}
        
        result_data, result_interpipe = self.block.transform(data, interpipe)
        
        self.assertIsInstance(result_data, tuple)
        self.assertEqual(len(result_data), 2)
        np.testing.assert_array_equal(result_data[0], data['neural'])
        np.testing.assert_array_equal(result_data[1], data['behaviour'])
        
    def test_transform_four_keys(self):
        data = {
            'neural_train': np.random.randn(80, 10),
            'neural_test': np.random.randn(20, 10),
            'behaviour_train': np.random.randn(80, 3),
            'behaviour_test': np.random.randn(20, 3)
        }
        interpipe = {}
        
        result_data, result_interpipe = self.block.transform(data, interpipe)
        
        self.assertIsInstance(result_data, tuple)
        self.assertEqual(len(result_data), 4)
        np.testing.assert_array_equal(result_data[0], data['neural_train'])
        np.testing.assert_array_equal(result_data[1], data['neural_test'])
        np.testing.assert_array_equal(result_data[2], data['behaviour_train'])
        np.testing.assert_array_equal(result_data[3], data['behaviour_test'])
        
    def test_invalid_key_count(self):
        data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        interpipe = {}
        
        with self.assertRaises(ValueError):
            self.block.transform(data, interpipe)

class TestDataset2DictBlock(unittest.TestCase):
    def setUp(self):
        self.block = Dataset2DictBlock(
            neural_nwb_loc='neural_path',
            behavior_nwb_loc='behavior_path', 
            time_nwb_loc='time_path'
        )
        
    def test_transform(self):
        class MockDataset:
            def __init__(self):
                self.dataset = {
                    'neural_path': np.random.randn(100, 10),
                    'behavior_path': np.random.randn(100, 3),
                    'time_path': np.arange(100)
                }
        
        from neuraldecoding.utils.utils_general import resolve_path
        original_func = resolve_path
        
        def mock_resolve_path(dataset, path):
            class MockArray:
                def __init__(self, data):
                    self.data = data
                def __getitem__(self, slice_obj):
                    return self.data
            return MockArray(dataset[path])
        
        import neuraldecoding.utils.utils_general
        neuraldecoding.utils.utils_general.resolve_path = mock_resolve_path
        
        try:
            mock_data = MockDataset()
            interpipe = {}
            
            result_data, result_interpipe = self.block.transform(mock_data, interpipe)
            
            self.assertIn('neural', result_data)
            self.assertIn('behaviour', result_data)
            self.assertIn('time_stamps', result_interpipe)
            
            np.testing.assert_array_equal(result_data['neural'], mock_data.dataset['neural_path'])
            np.testing.assert_array_equal(result_data['behaviour'], mock_data.dataset['behavior_path'])
            np.testing.assert_array_equal(result_interpipe['time_stamps'], mock_data.dataset['time_path'])
        finally:
            neuraldecoding.utils.utils_general.resolve_path = original_func

if __name__ == "__main__":
    unittest.main()
