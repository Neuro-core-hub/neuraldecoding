import neuraldecoding.utils
import neuraldecoding.stabilization.latent_space_alignment
from abc import ABC, abstractmethod

class PreprocessingWrapper(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, data, params = None):
        pass

class Dict2DataWrapper(PreprocessingWrapper):
    def __init__(self, neural_type = "sbp"):
        super().__init__()
        self.neural_type = neural_type

    def transform(self, data: dict, params = None):
        processed_data = neuraldecoding.utils.neural_finger_from_dict(data, self.neural_type)
        return processed_data

class StabilizationWrapper(PreprocessingWrapper):
    def __init__(self, stab_config):
        super().__init__()
        stabilization_method = getattr(neuraldecoding.stabilization.latent_space_alignment, stab_config["type"])
        self.stabilization = stabilization_method(stab_config["params"])

    def transform(self, data: tuple, params):
        neural, finger = data
        if 'is_train' not in params:
            raise ValueError("The 'params' dictionary for StabilizationWrapper must contain an 'is_train' key.")
        if params['is_train']:
            processed_data = (self.stabilization.fit(neural), finger)
            self.stabilization.save_alignment()
        else:
            self.stabilization.load_alignment()
            processed_data = (self.stabilization.extract_latent_space(neural), finger)
        return processed_data

class DataSplitWrapper(PreprocessingWrapper):
    def __init__(self, split_ratio=0.8):
        super().__init__()
        raise NotImplementedError("Not Implemented.")
    def transform(self, data, params=None):
        raise NotImplementedError("Not Implemented.")
    
class NormalizationWraper(PreprocessingWrapper):
    def __init__(self, split_ratio=0.8):
        super().__init__()
        raise NotImplementedError("Not Implemented.")
    def transform(self, data, params=None):
        raise NotImplementedError("Not Implemented.")
    
class AddHistoryWrapper(PreprocessingWrapper):
    def __init__(self, split_ratio=0.8):
        super().__init__()
        raise NotImplementedError("Not Implemented.")
    def transform(self, data, params=None):
        raise NotImplementedError("Not Implemented.")
    
class EnforceTensorWrapper(PreprocessingWrapper):
    def __init__(self, split_ratio=0.8):
        super().__init__()
        raise NotImplementedError("Not Implemented.")
    def transform(self, data, params=None):
        raise NotImplementedError("Not Implemented.")