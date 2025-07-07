# Import key modules and make them available at the package level
from neuraldecoding.feature_extraction import MultiModalFeatureExtractor
from neuraldecoding.filter import GenericFilter

__all__ = [
    'MultiModalFeatureExtractor',
    'GenericFilter',
]
