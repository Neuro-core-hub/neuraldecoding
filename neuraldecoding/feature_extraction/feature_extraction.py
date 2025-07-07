import numpy as np
import threading
from typing import Dict, List, Optional, Tuple, Union

class MultiModalFeatureExtractor:
    """
    Multi-modal feature extractor for neural data and kinematics.
    
    This class handles high-rate neural data and lower-rate kinematic data,
    binning them using the same time windows and extracting features from each modality.
    
    Attributes:
        bin_size_ms: Size of the time bin in milliseconds
        neural_config: Configuration for neural data processing
        kinematic_config: Configuration for kinematic data processing
        max_samples_per_bin: Maximum expected samples per bin (for buffer sizing)
        
    Performance optimizations:
        - Ring buffers to avoid memory allocations
        - Vectorized NumPy operations
        - Pre-computed indices for fast lookups
        - Thread-safe operations with minimal locking
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
        self.inv_bin_size_ms = 1.0 / self.bin_size_ms
        
        # Neural configuration
        self.neural_feature_type = self.neural_config.get('feature', 'mav')
        self.n_neural_channels = self.neural_config.get('channels', 0)
        self.neural_rate_hz = self.neural_config.get('expected_rate_hz', 1000)
        
        # Kinematic configuration
        self.kinematic_feature_type = self.kinematic_config.get('feature', 'mean')
        self.dof_names = self.kinematic_config.get('dofs', [])
        self.kinematic_rate_hz = self.kinematic_config.get('expected_rate_hz', 200)
        
        # Create DOF name to index mapping for fast lookups
        self.dof_to_index = {name: idx for idx, name in enumerate(self.dof_names)}
        self.n_dofs = len(self.dof_names)
        
        # Calculate buffer sizes (generous sizing to handle bursts)
        neural_samples_per_bin = int(self.neural_rate_hz * self.bin_size_sec * 1.5)  # 50% buffer
        kinematic_samples_per_bin = int(self.kinematic_rate_hz * self.bin_size_sec * 1.5)
        
        # Use the larger of the two for unified timestamp handling
        self.max_samples_per_bin = max(neural_samples_per_bin, kinematic_samples_per_bin, 100)
        
        # Ring buffers - pre-allocated for performance
        if self.n_neural_channels > 0:
            self.neural_buffer = np.zeros((self.max_samples_per_bin, self.n_neural_channels), dtype=np.float32)
            self.neural_timestamps = np.zeros(self.max_samples_per_bin, dtype=np.float64)
            self.neural_write_idx = 0
            self.neural_count = 0
        
        if self.n_dofs > 0:
            self.kinematic_buffer = np.zeros((self.max_samples_per_bin, self.n_dofs), dtype=np.float32)
            self.kinematic_timestamps = np.zeros(self.max_samples_per_bin, dtype=np.float64)
            self.kinematic_write_idx = 0
            self.kinematic_count = 0
        
        # Bin tracking
        self.current_bin_start = None
        self.last_bin_end = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        valid_features = ['mav', 'power', 'mean', 'var']
        
        if self.neural_feature_type not in valid_features:
            raise ValueError(f"Invalid neural feature type: {self.neural_feature_type}")
        
        if self.kinematic_feature_type not in valid_features:
            raise ValueError(f"Invalid kinematic feature type: {self.kinematic_feature_type}")
        
        if self.bin_size_ms <= 0:
            raise ValueError("bin_size_ms must be positive")
        
        if self.n_neural_channels == 0 and self.n_dofs == 0:
            raise ValueError("Must have at least neural channels or DOFs configured")
    
    
    def push_neural(self, samples: np.ndarray, timestamp: float) -> Optional[Dict[str, float]]:
        """
        Add neural data sample to the buffer.
        
        Args:
            samples: Neural data array of shape [samples, channels] or [channels]
            timestamp: Timestamp in milliseconds
            
        Returns:
            Features dict if a bin was completed, None otherwise
        """
        if self.n_neural_channels == 0:
            return None
        
        with self._lock:
            
            # Initialize bin tracking on first sample
            if self.current_bin_start is None:
                self.current_bin_start = timestamp
                self.last_bin_end = timestamp
            
            # Handle both single sample and multiple samples
            if samples.ndim == 1:
                # Single sample case
                samples_to_add = samples.reshape(1, -1)
            else:
                # Multiple samples case
                samples_to_add = samples
            
            # Add samples to ring buffer efficiently
            n_samples = samples_to_add.shape[0]
            
            # Calculate indices for batch insertion
            start_idx = self.neural_write_idx % self.max_samples_per_bin
            end_idx = (self.neural_write_idx + n_samples) % self.max_samples_per_bin
            
            if start_idx + n_samples <= self.max_samples_per_bin:
                # No wraparound - simple slice assignment
                self.neural_buffer[start_idx:start_idx + n_samples] = samples_to_add.astype(np.float32)
                self.neural_timestamps[start_idx:start_idx + n_samples] = timestamp
            else:
                # Handle wraparound
                first_chunk_size = self.max_samples_per_bin - start_idx
                self.neural_buffer[start_idx:] = samples_to_add[:first_chunk_size].astype(np.float32)
                self.neural_timestamps[start_idx:] = timestamp
                
                if end_idx > 0:
                    self.neural_buffer[:end_idx] = samples_to_add[first_chunk_size:].astype(np.float32)
                    self.neural_timestamps[:end_idx] = timestamp
            
            self.neural_write_idx += n_samples
            self.neural_count = min(self.neural_count + n_samples, self.max_samples_per_bin)
    
    def push_kinematic(self, dof_dict: Dict[str, float], timestamp: float) -> Optional[Dict[str, float]]:
        """
        Add kinematic data sample to the buffer.
        
        Args:
            dof_dict: Dictionary mapping DOF names to values
            timestamp: Timestamp in milliseconds
            
        Returns:
            Features dict if a bin was completed, None otherwise
        """
        if self.n_dofs == 0:
            return None
        
        with self._lock:
            # Initialize bin tracking on first sample
            if self.current_bin_start is None:
                self.current_bin_start = timestamp
                self.last_bin_end = timestamp
            
            # Write directly to buffer using pre-computed indices
            idx = self.kinematic_write_idx % self.max_samples_per_bin
            # Initialize buffer row to zero first
            self.kinematic_buffer[idx] = 0.0
            # Fill in the values for DOFs that are present
            for dof_name, value in dof_dict.items():
                if dof_name in self.dof_to_index:
                    self.kinematic_buffer[idx, self.dof_to_index[dof_name]] = float(value)
            self.kinematic_timestamps[idx] = timestamp
            
            self.kinematic_write_idx += 1
            self.kinematic_count = min(self.kinematic_count + 1, self.max_samples_per_bin)
    
    
    def extract_and_reset_bin(self, timestamp: float) -> Optional[Dict[str, float]]:
        """
        Extract features from the current bin and reset for the next bin.
        
        Args:
            timestamp: Current timestamp in milliseconds (used for bin metadata)
            
        Returns:
            Features dict if there's data to extract, None otherwise
        """
        # Initialize bin tracking on first call
        if self.current_bin_start is None:
            self.current_bin_start = timestamp
            self.last_bin_end = timestamp
            return None  # No data to extract yet
        
        # Check if we have any data to extract
        has_data = False
        if self.n_neural_channels > 0 and self.neural_count > 0:
            has_data = True
        if self.n_dofs > 0 and self.kinematic_count > 0:
            has_data = True
            
        if not has_data:
            return None
        
        # Extract features from current bin
        features = self._extract_bin_features(timestamp)
        
        # Reset for next bin
        self.current_bin_start = timestamp
        self.last_bin_end = timestamp
        
        # Reset counters but keep buffers (ring buffer reuse)
        if self.n_neural_channels > 0:
            self.neural_count = 0
            self.neural_write_idx = 0
        
        if self.n_dofs > 0:
            self.kinematic_count = 0
            self.kinematic_write_idx = 0
        
        return features
    
    def _extract_bin_features(self, bin_end_timestamp: float) -> Dict[str, float]:
        """
        Extract features from the current bin data.
        
        Args:
            bin_end_timestamp: Timestamp when bin was completed
            
        Returns:
            Dictionary mapping feature names to values
        """
        features = {}
        
        # Extract neural features
        if self.n_neural_channels > 0 and self.neural_count > 0:
            # Get valid samples from ring buffer
            if self.neural_count < self.max_samples_per_bin:
                # Simple case - no wraparound
                neural_data = self.neural_buffer[:self.neural_count]
            else:
                # Ring buffer wrapped - reconstruct proper order
                start_idx = self.neural_write_idx % self.max_samples_per_bin
                neural_data = np.concatenate([
                    self.neural_buffer[start_idx:],
                    self.neural_buffer[:start_idx]
                ])
            
            # Compute features vectorized
            neural_features = self._compute_features(neural_data, self.neural_feature_type)
            features["neural"] = neural_features
            features["neural_count"] = self.neural_count
            features["neural_feature_type"] = self.neural_feature_type
        # Extract kinematic features
        if self.n_dofs > 0 and self.kinematic_count > 0:
            # Get valid samples from ring buffer
            if self.kinematic_count < self.max_samples_per_bin:
                # Simple case - no wraparound
                kinematic_data = self.kinematic_buffer[:self.kinematic_count]
            else:
                # Ring buffer wrapped - reconstruct proper order
                start_idx = self.kinematic_write_idx % self.max_samples_per_bin
                kinematic_data = np.concatenate([
                    self.kinematic_buffer[start_idx:],
                    self.kinematic_buffer[:start_idx]
                ])
            
            # Compute features vectorized
            kinematic_features = self._compute_features(kinematic_data, self.kinematic_feature_type)
            # Convert back to dof names
            kinematic_dict = {}
            for dof_idx, dof_name in enumerate(self.dof_names):
                kinematic_dict[dof_name] = kinematic_features[dof_idx]
            features["kinematics"] = kinematic_dict
            features["kinematics_count"] = self.kinematic_count
            features["kinematics_feature_type"] = self.kinematic_feature_type
        
        # Add metadata
        features['bin_end_timestamp'] = bin_end_timestamp
        features['bin_start_timestamp'] = self.current_bin_start
        
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
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def reset(self):
        """Reset the internal state of the feature extractor."""
        with self._lock:
            # Reset buffers
            if self.n_neural_channels > 0:
                self.neural_buffer.fill(0)
                self.neural_timestamps.fill(0)
                self.neural_write_idx = 0
                self.neural_count = 0
            
            if self.n_dofs > 0:
                self.kinematic_buffer.fill(0)
                self.kinematic_timestamps.fill(0)
                self.kinematic_write_idx = 0
                self.kinematic_count = 0
            
            # Reset bin tracking
            self.current_bin_start = None
            self.last_bin_end = None


# Example usage
if __name__ == "__main__":
    # Test neural-only configuration (backward compatibility)
    neural_config = {
        'bin_size_ms': 100,
        'neural': {
            'feature': 'mav',
            'channels': 3,
            'expected_rate_hz': 1000
        }
    }
    
    extractor = MultiModalFeatureExtractor(neural_config)
    
    # Simulate neural data
    np.random.seed(42)
    features_list = []
    
    for i in range(50):
        timestamp = i * 10  # 10ms intervals
        neural_data = np.random.randn(3)
        
        features = extractor.push_neural(neural_data, timestamp)
        if features is not None:
            print(f"Time: {timestamp}ms, Features: {features}")
            features_list.append(features)
    
    print(f"\nNeural-only test - Total bins: {len(features_list)}")
    
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
    
    # Simulate mixed data streams
    features_list = []
    np.random.seed(42)
    
    for i in range(100):
        timestamp = i * 5  # 5ms intervals
        
        # Neural data every 1ms (simulated at 5ms for demo)
        neural_data = np.random.randn(4)
        features = mm_extractor.push_neural(neural_data, timestamp)
        
        # Kinematic data every 5ms (200Hz)
        if i % 1 == 0:  # Every sample for demo
            kinematic_data = {
                'thumb': np.random.uniform(0, 1),
                'index': np.random.uniform(0, 1),
                'middle': np.random.uniform(0, 1)
            }
            features = mm_extractor.push_kinematic(kinematic_data, timestamp)
        
        if features is not None:
            print(f"Time: {timestamp}ms, Multimodal Features: {len(features)} features")
            features_list.append(features)
    
    print(f"\nMultimodal test - Total bins: {len(features_list)}")
    
    # Show a sample feature vector
    if features_list:
        print(f"\nSample feature vector keys: {list(features_list[0].keys())}")
