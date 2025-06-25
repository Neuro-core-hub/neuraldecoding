import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class FeatureExtractor:
    """
    Class for extracting features from neural data.
    
    Attributes:
        feature_type: Type of feature to extract ('mav' for mean absolute value or 'power')
        bin_size_ms: Size of the time bin in milliseconds (default: 50ms)
        buffer: Buffer to store samples until a bin is complete
        buffer_timestamps: List of timestamps corresponding to samples in the buffer
        last_bin_end: Timestamp of the end of the last completed bin
    """
    
    def __init__(self, params: Dict):
        """
        Initialize the feature extractor with parameters.
        
        Args:
            params: Dictionary containing parameters for feature extraction
                   - feature_type: 'mav' for mean absolute value or 'power'
                   - bin_size_ms: Size of the time bin in milliseconds (default: 50ms)
        """
        self.feature_type = params.get('feature_type', 'mav')
        self.bin_size_ms = params.get('bin_size_ms', 50)
        
        # Initialize buffer and time tracking
        self.buffer = []
        self.buffer_timestamps = []
        self.last_bin_end = None
    
    def extract_feature(self, samples: np.ndarray) -> np.ndarray:
        """
        Extract features from a complete bin of samples.
        
        Args:
            samples: Numpy array containing a complete bin of samples [samples, channels]
            
        Returns:
            Extracted features as a numpy array
        """
        if self.feature_type == 'mav':
            # Mean Absolute Value (MAV)
            return np.mean(np.abs(samples), axis=0)
        elif self.feature_type == 'power':
            # Power (mean of squared values)
            return np.mean(np.square(samples), axis=0)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def apply(self, data: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """
        Apply feature extraction to incoming data.
        
        Args:
            data: Numpy array of shape [channels] containing samples for all channels
            timestamp: Time value in milliseconds
            
        Returns:
            Extracted features when a bin is complete, None otherwise
        """
        # Initialize last_bin_end if this is the first call
        if self.last_bin_end is None:
            self.last_bin_end = timestamp
        
        # Add data and timestamp to the buffer
        self.buffer.append(data)
        self.buffer_timestamps.append(timestamp)
        
        # Check if we have collected enough time for a complete bin
        elapsed_time = timestamp - self.last_bin_end
        
        if elapsed_time >= self.bin_size_ms:
            # Convert buffer to numpy array
            bin_data = np.array(self.buffer)
            
            # Extract features from the bin
            features = self.extract_feature(bin_data)
            
            # Update the last bin end time
            self.last_bin_end = timestamp
            
            # Reset buffer
            self.buffer = []
            self.buffer_timestamps = []
            
            return features
        
        return None
    
    def reset(self):
        """Reset the internal state of the feature extractor."""
        self.buffer = []
        self.buffer_timestamps = []
        self.last_bin_end = None
        
    def get_description(self) -> str:
        """
        Get a formatted description of the feature extractor for logging.
        
        Returns:
            A string describing the feature extractor configuration
        """
        feature_name = {
            'mav': 'Mean Absolute Value',
            'power': 'Power'
        }.get(self.feature_type, self.feature_type)
        
        return (f"FeatureExtractor: {feature_name} | "
                f"Bin Size: {self.bin_size_ms}ms")


# Example usage
if __name__ == "__main__":
    # Create a feature extractor for MAV with a 100ms bin size
    feature_params = {
        'feature_type': 'mav',
        'bin_size_ms': 100
    }
    extractor = FeatureExtractor(feature_params)
    
    # Print the extractor description
    print(extractor.get_description())
    
    # Simulate incoming neural data (3 channels)
    # In a real application, this would come from a neural recording device
    np.random.seed(42)
    
    # Process 500ms of data with samples every 10ms
    features_list = []
    for i in range(50):
        # Generate random neural data with 3 channels
        timestamp = i * 10  # Timestamp in ms
        neural_data = np.random.randn(3)  # 3 channels of neural data
        
        # Apply feature extraction
        features = extractor.apply(neural_data, timestamp)
        
        # If features were extracted (a bin was completed), print them
        if features is not None:
            print(f"Time: {timestamp}ms, Features: {features}")
            features_list.append(features)
    
    # Print the number of feature vectors extracted
    print(f"\nTotal number of feature vectors: {len(features_list)}")
    
    # Create a feature extractor for power with a 50ms bin size
    power_params = {
        'feature_type': 'power',
        'bin_size_ms': 50
    }
    power_extractor = FeatureExtractor(power_params)
    
    # Print the power extractor description
    print(power_extractor.get_description())
    
    # Reset feature list
    features_list = []
    
    # Process the same data using power features
    for i in range(50):
        timestamp = i * 10
        neural_data = np.random.randn(3)
        
        features = power_extractor.apply(neural_data, timestamp)
        if features is not None:
            print(f"Time: {timestamp}ms, Power Features: {features}")
            features_list.append(features)
    
    print(f"\nTotal number of power feature vectors: {len(features_list)}")
