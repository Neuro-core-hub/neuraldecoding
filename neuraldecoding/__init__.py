# Import key modules and make them available at the package level
from .feature_extraction import FeatureExtractor
from .filter import GenericFilter

__all__ = [
    'FeatureExtractor',
    'GenericFilter',
]
