import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class FeatureExtractor:
    """
    Generic feature extractor for time-series data.
    
    This class processes batches of multi-dimensional data for time bins,
    extracting features from each time window.
    
    Attributes:
        bin_size_ms: Size of the time bin in milliseconds
        feature_type: Type of feature to extract ('mav', 'power', 'mean', 'var', 'mean_and_vel')
        channels: Expected number of dimensions/channels in the data
        expected_rate_hz: Expected sampling rate in Hz
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary with structure:
                {
                    "bin_size_ms": 50,
                    "feature_type": "mav",  # 'mav', 'power', 'mean', 'var', 'mean_and_vel'
                    "channels": 96,  # number of channels/features
                    "expected_rate_hz": 1000
                }
        """
        self.bin_size_ms = config.get('bin_size_ms', 50)
        self.feature_type = config.get('feature_type', 'mav')
        self.channels = config.get('channels', 1)
        self.expected_rate_hz = config.get('expected_rate_hz', 1000)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        valid_features = ['mav', 'power', 'mean', 'var', 'mean_and_vel']
        
        if self.feature_type not in valid_features:
            raise ValueError(f"Invalid feature type: {self.feature_type}. Must be one of {valid_features}")
        
        if self.bin_size_ms <= 0:
            raise ValueError("bin_size_ms must be positive")
        
        if self.channels <= 0:
            raise ValueError("channels must be positive")
    
    def extract_binned_features(self,
                               data: np.ndarray,
                               timestamps_ms: np.ndarray) -> List[Dict]:
        """
        Extract features from timestamped data by dividing it into time bins.
        
        Args:
            data: Data array of shape [n_samples, dimensions]
            timestamps_ms: Timestamps for data of shape [n_samples]
            
        Returns:
            List of feature dictionaries, one per bin
        """
        # Validate inputs
        if data.shape[0] != timestamps_ms.shape[0]:
            raise ValueError("Data and timestamps must have same number of samples")
        
        if len(timestamps_ms) == 0:
            return []
        
        # Determine time range
        min_time = timestamps_ms.min()
        max_time = timestamps_ms.max()
        
        # Create bins
        bin_features = []
        current_time = min_time
        
        while current_time <= max_time:
            bin_start = current_time
            bin_end = current_time + self.bin_size_ms
            
            # Extract samples for this bin
            bin_mask = (timestamps_ms >= bin_start) & (timestamps_ms < bin_end)
            
            if np.any(bin_mask):
                bin_data = data[bin_mask]
                
                # Extract features for this bin
                features = self.compute_bin_features(
                    data=bin_data,
                    bin_end_timestamp_ms=bin_end
                )
                
                # Add bin timing information
                if features is not None:
                    features['bin_start_ms'] = bin_start
                    features['bin_end_ms'] = bin_end
                    features['bin_center_ms'] = (bin_start + bin_end) / 2.0
                    bin_features.append(features)
            
            current_time += self.bin_size_ms
        
        return bin_features

    def compute_bin_features(self, 
                            data: np.ndarray,
                            bin_end_timestamp_ms: Optional[float] = None) -> Optional[Dict]:
        """
        Extract features from a single bin of data.
        
        Args:
            data: Data array of shape [n_samples, dimensions] or [dimensions] for single sample
            bin_end_timestamp_ms: Timestamp at the end of this bin (milliseconds)
            
        Returns:
            Dictionary containing extracted features, or None if no data
        """
        if data is None or data.size == 0:
            return None
        
        # Handle both single sample and multiple samples
        if data.ndim == 1:
            # Single sample case - reshape to [1, dimensions]
            processed_data = data.reshape(1, -1)
        else:
            # Multiple samples case - use as is
            processed_data = data
        
        # Throw warning if the number of channels is greater than the expected number of channels
        if processed_data.shape[1] > self.channels:
            self.logger.warning(f"Number of channels in data ({processed_data.shape[1]}) is greater than the expected number of channels ({self.channels})")
            processed_data = processed_data[:, :self.channels]
        
        # Compute features
        features_array = self._compute_features(processed_data, self.feature_type)
        
        # Create result dictionary
        features = {
            "features": features_array,
            "sample_count": processed_data.shape[0],
            "feature_type": self.feature_type,
            "dimensions": processed_data.shape[1]
        }
        
        # Add metadata
        if bin_end_timestamp_ms is not None:
            features['bin_end_timestamp_ms'] = bin_end_timestamp_ms
            features['bin_start_timestamp_ms'] = bin_end_timestamp_ms - self.bin_size_ms
        
        return features
    
    def _compute_features(self, data: np.ndarray, feature_type: str) -> np.ndarray:
        """
        Compute features from data using vectorized operations.
        
        Args:
            data: Input data of shape [samples, channels]
            feature_type: Type of feature to compute
            
        Returns:
            Feature array of shape [channels] or [2*channels] for mean_and_vel
        """
        if data.shape[0] == 0:
            return np.zeros(data.shape[1], dtype=np.float32)
        
        if feature_type == 'mav':
            # Mean Absolute Value
            return np.mean(np.abs(data), axis=0)
        elif feature_type == 'power':
            # Power (mean of squared values)
            return np.mean(np.square(data), axis=0)
        elif feature_type == 'mean':
            # Arithmetic mean
            return np.mean(data, axis=0)
        elif feature_type == 'var':
            # Variance
            return np.var(data, axis=0)
        elif feature_type == "mean_and_vel":
            # Mean and velocity
            mean_pos = np.mean(data, axis=0)
            vel = np.diff(data, axis=0).mean(axis=0) if data.shape[0] > 1 else np.zeros(data.shape[1])
            return np.concatenate((mean_pos, vel))
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    

# Example usage
if __name__ == "__main__":
    # Test basic configuration
    config = {
        'bin_size_ms': 100,
        'feature_type': 'mav',
        'channels': 3,
        'expected_rate_hz': 1000
    }
    
    extractor = FeatureExtractor(config)
    
    # Simulate data batch
    np.random.seed(42)
    data_batch = np.random.randn(10, 3)  # 10 samples, 3 dimensions
    
    features = extractor.compute_bin_features(
        data=data_batch,
        bin_end_timestamp_ms=100.0
    )
    
    print(f"Single bin features: {features}")
    
    # Test timestamped data processing
    print("\n--- Testing timestamped data processing ---")
    
    # Create timestamped data (1000 Hz)
    timestamps = np.arange(0, 500, 1)  # 0-500ms, 1ms intervals
    data = np.random.randn(len(timestamps), 4)  # 4 dimensions
    
    # Update config for 4 dimensions
    config['channels'] = 4
    extractor = FeatureExtractor(config)
    
    # Extract features per bin
    bin_features = extractor.extract_binned_features(
        data=data,
        timestamps_ms=timestamps
    )
    
    print(f"Number of bins created: {len(bin_features)}")
    print(f"First bin features: {bin_features[0] if bin_features else 'No bins created'}")
    print(f"Last bin features: {bin_features[-1] if bin_features else 'No bins created'}")
    
    # Test different feature types
    print("\n--- Testing different feature types ---")
    feature_types = ['mav', 'power', 'mean', 'var', 'mean_and_vel']
    
    for ft in feature_types:
        config['feature_type'] = ft
        extractor = FeatureExtractor(config)
        
        features = extractor.compute_bin_features(
            data=data_batch,
            bin_end_timestamp_ms=100.0
        )
        
        print(f"{ft} features shape: {features['features'].shape}")
