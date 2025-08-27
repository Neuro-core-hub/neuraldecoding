# Import key modules and make them available at the package level
from neuraldecoding.feature_extraction import FeatureExtractor
from neuraldecoding.filter import GenericFilter
from neuraldecoding.feature_extraction import FeatureExtractor
from neuraldecoding.filter import GenericFilter
from neuraldecoding.dataset import Dataset
from neuraldecoding.preprocessing import Preprocessing
from neuraldecoding.trainer import LinearTrainer

__all__ = [
    'FeatureExtractor',
    'GenericFilter',
    'Dataset',
    'Preprocessing',
    'LinearTrainer',
]
