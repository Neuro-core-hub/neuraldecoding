# Import key modules and make them available at the package level
from .feature_extraction import FeatureExtractor
from .filter import GenericFilter
from .feature_extraction import FeatureExtractor
from .filter import GenericFilter
from .dataset import Dataset
from .preprocessing import Preprocessing
from .trainer import LinearTrainer

__all__ = [
    'FeatureExtractor',
    'GenericFilter',
    'Dataset',
    'Preprocessing',
    'LinearTrainer',
]
