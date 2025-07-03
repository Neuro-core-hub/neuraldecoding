import neuraldecoding.utils
import neuraldecoding.stabilization.latent_space_alignment
import neuraldecoding.dataaugmentation.DataAugmentation

import sklearn.preprocessing

import torch

from abc import ABC, abstractmethod

import time

class PreprocessingWrapper(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, data, interpipe, params = None):
        pass

# Wrappers that Modifies Data Format
class Dict2DataDictWrapper(PreprocessingWrapper):
    """
    Converts a dictionary (from nwb) to a neural and finger data in dictionary format.
    neural data type can be specified by the user.
    Wrapper that modifies the data format.
    """
    def __init__(self, neural_type = "sbp"):
        super().__init__()
        self.neural_type = neural_type

    def transform(self, data: dict, interpipe, params = None):
        (neural, finger), trial_idx = neuraldecoding.utils.neural_finger_from_dict(data, self.neural_type)
        interpipe['trial_idx'] = trial_idx

        data_out = {'neural': neural, 'finger': finger}
        return data_out, interpipe

class DataSplitWrapper(PreprocessingWrapper):
    def __init__(self, split_ratio: 0.8, split_seed: 42):
        super().__init__()
        self.split_ratio = split_ratio
        self.split_seed = split_seed

    def transform(self, data, interpipe, params=None):
        if 'trial_idx' not in interpipe:
            raise ValueError("DataSplitWrapper requires 'trial_idx' in interpipe from other wrappers (Dict2DataWrapper).")
        
        split_data = neuraldecoding.utils.data_split_trial(data['neural'], 
                                                           data['finger'], 
                                                           interpipe['trial_idx'], 
                                                           split_ratio=self.split_ratio, 
                                                           seed=self.split_seed)
        
        (neural_train, finger_train), (neural_test, finger_test) = split_data
        data_out = {'neural_train': neural_train, 
                    'neural_test': neural_test, 
                    'finger_train': finger_train, 
                    'finger_test': finger_test}
        return data_out, interpipe

class Dict2TupleWrapper(PreprocessingWrapper):
    def __init__(self):
        super().__init__()

    def transform(self, data, interpipe, params=None):
        if len(data) == 2:
            data_out = (data['neural'] , data['finger'])
        elif len(data) == 4:
            data_out = (data['neural_train'], data['neural_test'], data['finger_train'], data['finger_test'])
        else:
            raise ValueError(f"Data Dict Contain Unexpected # of Keys. Expected 2 or 4 keys, got {len(data)}")
        return data_out, interpipe

# Wrappers that Modify Data
class StabilizationWrapper(PreprocessingWrapper):
    def __init__(self, location, stabilization_config):
        super().__init__()
        stabilization_method = getattr(neuraldecoding.stabilization.latent_space_alignment, stabilization_config["type"])
        self.stabilization = stabilization_method(stabilization_config["params"])
        self.location = location

    def transform(self, data, interpipe, params):
        if 'is_train' not in params:
            raise ValueError("The 'params' dictionary for StabilizationWrapper must contain an 'is_train' key.")

        if params['is_train']:
            data[self.location] = self.stabilization.fit(data[self.location])
            self.stabilization.save_alignment()
        else:
            self.stabilization.load_alignment()
            data[self.location] = self.stabilization.extract_latent_space(data[self.location])
        return data, interpipe

class AddHistoryWrapper(PreprocessingWrapper):
    def __init__(self, location, seq_length = 10):
        super().__init__()
        self.location = location
        self.seq_length = seq_length

    def transform(self, data, interpipe, params=None):
        if isinstance(self.location, str):
            self.location = [self.location]

        for loc in self.location:
            data[loc] = neuraldecoding.utils.add_history_numpy(data[loc], self.seq_length)

        return data, interpipe
    
class NormalizationWraper(PreprocessingWrapper):
    def __init__(self, location, method, normalizer_params):
        super().__init__()
        self.location = location
        self.normalizer_method = method
        self.normalizer_params = normalizer_params

    def transform(self, data, interpipe, params=None):
        if self.normalizer_method == 'moving_average':
            p = self.normalizer_params['params']
        elif self.normalizer_method == 'sklearn':
            normalizer = getattr(sklearn.preprocessing, self.normalizer_params['type'])
            p = {'normalizer': normalizer(**self.normalizer_params['params'])}
        else:
            p = {}
        for loc in self.location:
            data[loc], _ = neuraldecoding.dataaugmentation.DataAugmentation.normalize(data[loc],
                                                                                   method = self.normalizer_method,
                                                                                   **p)
        return data, interpipe
                                                              
class EnforceTensorWrapper(PreprocessingWrapper):
    def __init__(self, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = getattr(torch, dtype)

    def transform(self, data, interpipe, params=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.device, dtype=self.dtype)
            else:
                data[key] = torch.tensor(data[key], device=self.device, dtype=self.dtype)
        return data, interpipe
