import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class MultiModalFeatureExtractor:
    """
    Multi-modal feature extractor for neural data and kinematics.
    
    This class processes batches of neural and kinematic data for time bins,
    extracting features from each modality.
    
    Attributes:
        bin_size_ms: Size of the time bin in milliseconds
        neural_config: Configuration for neural data processing
        kinematic_config: Configuration for kinematic data processing
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the multi-modal feature extractor.
        
        Args:
            config: Configuration dictionary with structure:
                {
                    "bin_size_ms": 50,
                    "neural": {
                        "feature": "mav",  # 'mav', 'power', 'mean', 'var'
                        "channels": 96,
                        "expected_rate_hz": 1000
                    },
                    "kinematic": {
                        "feature": "mean",  # 'mean', 'var', 'mav', 'power'
                        "dofs": ["thumb", "index", "middle", "ring", "pinky"],
                        "expected_rate_hz": 200
                    }
                }
        """
        self.bin_size_ms = config.get('bin_size_ms', 50)
        self.neural_config = config.get('neural', {})
        self.kinematic_config = config.get('kinematic', {})
        
        # Pre-compute for performance
        self.bin_size_sec = self.bin_size_ms / 1000.0
        
        # Neural configuration
        self.neural_feature_type = self.neural_config.get('feature', 'mav')
        self.n_neural_channels = self.neural_config.get('channels', 0)
        self.neural_rate_hz = self.neural_config.get('expected_rate_hz', 1000)
        
        # Kinematic configuration
        self.kinematic_feature_type = self.kinematic_config.get('feature', 'mean')
        self.dof_names = self.kinematic_config.get('dofs', [])
        self.kinematic_rate_hz = self.kinematic_config.get('expected_rate_hz', 200)
        
        # Create DOF name to index mapping for fast lookups
        self.n_dofs = len(self.dof_names)
        
        # Pre-allocate working arrays for performance (generous sizing for bursts)
        max_neural_samples = int(self.neural_rate_hz * self.bin_size_sec * 2.0)  # 2x buffer
        max_kinematic_samples = int(self.kinematic_rate_hz * self.bin_size_sec * 2.0)
        
        if self.n_neural_channels > 0:
            self.neural_work_buffer = np.zeros((max_neural_samples, self.n_neural_channels), dtype=np.float32)
        
        if self.n_dofs > 0:
            self.kinematic_work_buffer = np.zeros((max_kinematic_samples, self.n_dofs), dtype=np.float32)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        valid_features = ['mav', 'power', 'mean', 'var', 'mean_and_vel']
        
        if self.neural_feature_type not in valid_features:
            raise ValueError(f"Invalid neural feature type: {self.neural_feature_type}")
        
        if self.kinematic_feature_type not in valid_features:
            raise ValueError(f"Invalid kinematic feature type: {self.kinematic_feature_type}")
        
        if self.bin_size_ms <= 0:
            raise ValueError("bin_size_ms must be positive")
        
        if self.n_neural_channels == 0 and self.n_dofs == 0:
            raise ValueError("Must have at least neural channels or DOFs configured")
    
    def extract_features_per_bin(self,
                                neural: Optional[np.ndarray] = None,
                                neural_timestamps_ms: Optional[np.ndarray] = None,
                                kinematics: Optional[Dict[str, np.ndarray]] = None,
                                kinematics_timestamps_ms: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Extract features per bin from timestamped neural and kinematic data.
        
        Args:
            neural: Neural data array of shape [n_samples1, channels]
            neural_timestamps_ms: Timestamps for neural data of shape [n_samples1]
            kinematics: Dictionary of kinematic data arrays, each of shape [n_samples2]
            kinematics_timestamps_ms: Timestamps for kinematic data of shape [n_samples2]
            
        Returns:
            List of feature dictionaries, one per bin
        """
        # Validate inputs
        if neural is not None and neural_timestamps_ms is not None:
            if neural.shape[0] != neural_timestamps_ms.shape[0]:
                raise ValueError("Neural data and timestamps must have same number of samples")
        
        if kinematics is not None and kinematics_timestamps_ms is not None:
            for dof_name, dof_data in kinematics.items():
                if dof_data.shape[0] != kinematics_timestamps_ms.shape[0]:
                    raise ValueError(f"Kinematic data for {dof_name} and timestamps must have same number of samples")
        
        # Determine time range
        all_timestamps = []
        if neural_timestamps_ms is not None:
            all_timestamps.extend(neural_timestamps_ms)
        if kinematics_timestamps_ms is not None:
            all_timestamps.extend(kinematics_timestamps_ms)
        
        if not all_timestamps:
            return []
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # Create bins
        bin_features = []
        current_time = min_time
        
        while current_time <= max_time:
            bin_start = current_time
            bin_end = current_time + self.bin_size_ms
            
            # Extract neural samples for this bin
            neural_samples = None
            if neural is not None and neural_timestamps_ms is not None:
                neural_mask = (neural_timestamps_ms >= bin_start) & (neural_timestamps_ms < bin_end)
                if np.any(neural_mask):
                    neural_samples = neural[neural_mask]
            
            # Extract kinematic samples for this bin
            kinematic_samples = None
            if kinematics is not None and kinematics_timestamps_ms is not None:
                kinematic_mask = (kinematics_timestamps_ms >= bin_start) & (kinematics_timestamps_ms < bin_end)
                if np.any(kinematic_mask):
                    kinematic_samples = {}
                    for dof_name, dof_data in kinematics.items():
                        if dof_name in self.dof_names:  # Only include configured DOFs
                            kinematic_samples[dof_name] = dof_data[kinematic_mask]
            
            # Extract features for this bin using existing method
            features = self.extract_bin_features(
                neural_samples=neural_samples,
                kinematic_samples=kinematic_samples,
                bin_timestamp=bin_end  # Use bin end as timestamp
            )
            
            # Add bin timing information
            if features is not None:
                features['bin_start_ms'] = bin_start
                features['bin_end_ms'] = bin_end
                features['bin_center_ms'] = (bin_start + bin_end) / 2.0
                bin_features.append(features)
            
            current_time += self.bin_size_ms
        
        return bin_features

    def extract_bin_features(self, 
                           neural_samples: Optional[np.ndarray] = None,
                           kinematic_samples: Optional[Dict[str, np.ndarray]] = None,
                           bin_timestamp: Optional[float] = None) -> Optional[Dict]:
        """
        Extract features from batches of neural and kinematic data for a single bin.
        
        Args:
            neural_samples: Neural data array, each of shape [channels] or [samples, channels]
            kinematic_samples: Dictionary of kinematic data arrays, each of shape [samples]
            bin_timestamp: Timestamp for this bin (milliseconds)
            
        Returns:
            Dictionary containing extracted features, or None if no data
        """
        features = {}
        has_data = False
        
        # Process neural data
        if neural_samples is not None and self.n_neural_channels > 0:
            # Handle both single sample and multiple samples
            if neural_samples.ndim == 1:
                # Single sample case - reshape to [1, channels]
                neural_data = neural_samples.reshape(1, -1)
            else:
                # Multiple samples case - use as is
                neural_data = neural_samples
            
            # Ensure we don't exceed the configured number of channels
            if neural_data.shape[1] > self.n_neural_channels:
                neural_data = neural_data[:, :self.n_neural_channels]
            
            # Compute neural features
            neural_features = self._compute_features(neural_data, self.neural_feature_type)
            features["neural"] = neural_features
            features["neural_count"] = neural_data.shape[0]
            features["neural_feature_type"] = self.neural_feature_type
            has_data = True
        
        # Process kinematic data
        kinematic_features = {dof: None for dof in self.dof_names}
        if kinematic_samples is not None and self.n_dofs > 0:
            # Convert dictionary of arrays to single array format
            for dof_name, dof_values in kinematic_samples.items():
                feats = self._compute_features(dof_values, self.kinematic_feature_type)
                kinematic_features[dof_name] = feats
            
            features["kinematics"] = kinematic_features
            features["kinematics_count"] = kinematic_samples[self.dof_names[0]].shape[0]
            features["kinematics_feature_type"] = self.kinematic_feature_type
            has_data = True
        
        if not has_data:
            return None
        
        # Add metadata
        if bin_timestamp is not None:
            features['bin_timestamp'] = bin_timestamp
            features['bin_start_timestamp'] = bin_timestamp - self.bin_size_ms
        
        return features
    
    def _compute_features(self, data: np.ndarray, feature_type: str) -> np.ndarray:
        """
        Compute features from data using vectorized operations.
        
        Args:
            data: Input data of shape [samples, channels]
            feature_type: Type of feature to compute
            
        Returns:
            Feature array of shape [channels]
        """
        if data is None or data.shape[0] == 0:
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
    # Test neural-only configuration
    neural_config = {
        'bin_size_ms': 100,
        'neural': {
            'feature': 'mav',
            'channels': 3,
            'expected_rate_hz': 1000
        }
    }
    
    extractor = MultiModalFeatureExtractor(neural_config)
    
    # Simulate neural data batch
    np.random.seed(42)
    neural_batch = np.random.randn(10, 3)  # 10 samples, 3 channels
    
    features = extractor.extract_bin_features(
        neural_samples=neural_batch,
        bin_timestamp=100.0
    )
    
    print(f"Neural-only features: {features}")
    
    # Test multi-modal configuration
    multimodal_config = {
        'bin_size_ms': 50,
        'neural': {
            'feature': 'mav',
            'channels': 4,
            'expected_rate_hz': 1000
        },
        'kinematic': {
            'feature': 'mean',
            'dofs': ['thumb', 'index', 'middle'],
            'expected_rate_hz': 200
        }
    }
    
    mm_extractor = MultiModalFeatureExtractor(multimodal_config)
    
    # Simulate mixed data batch
    neural_batch = np.random.randn(5, 4)  # 5 samples, 4 channels
    kinematic_batch = {
        'thumb': np.random.uniform(0, 1, size=3),
        'index': np.random.uniform(0, 1, size=3),
        'middle': np.random.uniform(0, 1, size=3)
    }
    
    features = mm_extractor.extract_bin_features(
        neural_samples=neural_batch,
        kinematic_samples=kinematic_batch,
        bin_timestamp=50.0
    )
    
    print(f"Multimodal features: {features}")
    
    # Test timestamped data processing with extract_features_per_bin
    print("\n--- Testing timestamped data processing ---")
    
    # Create timestamped neural data (1000 Hz)
    neural_timestamps = np.arange(0, 500, 1)  # 0-500ms, 1ms intervals
    neural_data = np.random.randn(len(neural_timestamps), 4)
    
    # Create timestamped kinematic data (200 Hz)  
    kinematic_timestamps = np.arange(0, 500, 5)  # 0-500ms, 5ms intervals
    kinematic_data = {
        'thumb': np.random.uniform(0, 1, size=len(kinematic_timestamps)),
        'index': np.random.uniform(0, 1, size=len(kinematic_timestamps)),
        'middle': np.random.uniform(0, 1, size=len(kinematic_timestamps))
    }
    
    # Extract features per bin
    bin_features = mm_extractor.extract_features_per_bin(
        neural=neural_data,
        neural_timestamps_ms=neural_timestamps,
        kinematics=kinematic_data,
        kinematics_timestamps_ms=kinematic_timestamps
    )
    
    print(f"Number of bins created: {len(bin_features)}")
    print(f"First bin features: {bin_features[0] if bin_features else 'No bins created'}")
    print(f"Last bin features: {bin_features[-1] if bin_features else 'No bins created'}")
