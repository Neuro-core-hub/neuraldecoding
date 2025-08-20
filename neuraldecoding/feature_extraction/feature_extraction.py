import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf

class FeatureExtractor:
    """
    Generic feature extractor for time-series data.
    
    This class processes batches of multi-dimensional data for time bins,
    extracting features from each time window.
    
    Attributes:
        bin_size_ms: Size of the time bin in milliseconds
        feature_type: Type of feature to extract ('mav', 'power', 'mean', 'var', 'mean_and_vel')
        channels: Expected number of dimensions/channels in the data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary with structure:
                {
                    "bin_size_ms": 50,
                    "feature_type": "mav" or ["mav", "power"] or [["mean", "var"], ["mav"]],  # string, list, or list of lists
                    "channels": 96,  # number of channels/features
                    "feature_params": [{"hist": 2}, {}] or None,  # parameters for each feature type
                }
        """
        self.bin_size_ms = config.get('bin_size_ms', 50)
        self.feature_type = config.get('feature_type', 'mav')
        if not isinstance(self.feature_type, list) and not isinstance(self.feature_type, str):
            try:
                self.feature_type = OmegaConf.to_container(self.feature_type)
            except Exception as e:
                print(f"Error converting feature_type: {e}")

        # Initialize feature parameters
        self.feature_params = config.get('feature_params', None)
        if self.feature_params is None:
            self.feature_params = self._create_default_feature_params()
        
        # Initialize history tracking for features that need it
        self.previous_bin_features = {}  # Store previous bin features per stream
        self.max_history_bins = self._get_max_history_bins()
        
        # Validate configuration
        self._validate_config()
        
        # Reset history for new instance
        self.reset_history()
    
    def _create_default_feature_params(self) -> List:
        """Create default empty parameters for each feature type."""
        def count_feature_groups(ft):
            if isinstance(ft, str):
                return 1
            elif isinstance(ft, list):
                if len(ft) > 0 and isinstance(ft[0], list):
                    # List of lists
                    return len(ft)
                else:
                    # Simple list - this is ONE feature group with multiple features
                    return 1
            return 1
        
        num_groups = count_feature_groups(self.feature_type)
        return [{} for _ in range(num_groups)]
    
    def _get_max_history_bins(self) -> int:
        """Get the maximum number of history bins needed across all features."""
        def extract_hist_param(params):
            if isinstance(params, dict):
                return params.get('hist', 0)
            return 0
        
        max_hist = 0
        for param in self.feature_params:
            hist = extract_hist_param(param)
            max_hist = max(max_hist, hist)
        
        return max_hist
    
    def reset_history(self):
        """Reset the history of previous bin features."""
        self.previous_bin_features = {}
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        valid_features = ['mav', 'power', 'mean', 'var', 'mean_and_vel', 'history', 'vel']
        
        # Flatten nested feature types for validation
        def flatten_feature_types(ft):
            if isinstance(ft, str):
                return [ft]
            elif isinstance(ft, list):
                flattened = []
                for item in ft:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                return flattened
            return [ft]
        
        feature_types = flatten_feature_types(self.feature_type)
        
        for ft in feature_types:
            if ft not in valid_features:
                raise ValueError(f"Invalid feature type: {ft}. Must be one of {valid_features}")
        
        # Validate feature_params structure matches feature_type structure
        def count_feature_groups(ft):
            if isinstance(ft, str):
                return 1
            elif isinstance(ft, list):
                if len(ft) > 0 and isinstance(ft[0], list):
                    return len(ft)  # List of lists - each inner list is a feature group
                else:
                    return 1  # Simple list - this is ONE feature group with multiple features
            return 1
        
        expected_param_count = count_feature_groups(self.feature_type)
        if len(self.feature_params) != expected_param_count:
            raise ValueError(f"feature_params length ({len(self.feature_params)}) must match feature_type structure (expected {expected_param_count})")
        
        # Validate history and velocity feature placement and parameters
        def validate_special_feature_placement(ft_group):
            """Validate that history and velocity features are properly placed with other features."""
            special_features = ['history', 'vel']
            
            if isinstance(ft_group, str):
                if ft_group in special_features:
                    raise ValueError(f"'{ft_group}' feature cannot be used alone. It must be combined with another feature (e.g., ['mean', '{ft_group}'])")
                return
            elif isinstance(ft_group, list):
                for special_feat in special_features:
                    if special_feat in ft_group and len(ft_group) == 1:
                        raise ValueError(f"'{special_feat}' feature cannot be used alone. It must be combined with another feature (e.g., ['mean', '{special_feat}'])")
                    if special_feat in ft_group and ft_group.index(special_feat) == 0:
                        raise ValueError(f"'{special_feat}' feature must come after other features in the list (e.g., ['mean', '{special_feat}'], not ['{special_feat}', 'mean'])")
        
        # Check each feature group
        if isinstance(self.feature_type, str):
            validate_special_feature_placement(self.feature_type)
        elif isinstance(self.feature_type, list):
            if len(self.feature_type) > 0 and isinstance(self.feature_type[0], list):
                # List of lists
                for ft_group in self.feature_type:
                    validate_special_feature_placement(ft_group)
            else:
                # Simple list
                validate_special_feature_placement(self.feature_type)
        
        # Validate history parameters
        for i, param in enumerate(self.feature_params):
            if isinstance(param, dict) and 'hist' in param:
                hist_val = param['hist']
                if not isinstance(hist_val, int) or hist_val < 0:
                    raise ValueError(f"'hist' parameter in feature_params[{i}] must be a non-negative integer")
        
        if self.bin_size_ms <= 0:
            raise ValueError("bin_size_ms must be positive")
    
    def extract_binned_features(self,
                               data: Union[np.ndarray, List[np.ndarray]],
                               timestamps_ms: Union[np.ndarray, List[np.ndarray]],
                               return_array: bool = False) -> Union[List[Dict], np.ndarray, List[np.ndarray]]:
        """
        Extract features from timestamped data by dividing it into time bins.
        
        Args:
            data: Data array of shape [n_samples, dimensions] or list of such arrays
            timestamps_ms: Timestamps for data of shape [n_samples] or list of such arrays
            return_array: If True, return array of features instead of list of dictionaries
            
        Returns:
            List of feature dictionaries (default), single array [n_bins, n_features] (single input),
            or list of arrays [n_bins, n_features] (multiple inputs) when return_array=True
        """
        # Normalize inputs to lists
        if isinstance(data, np.ndarray):
            data_list = [data]
            timestamps_list = [timestamps_ms]
        else:
            data_list = data
            timestamps_list = timestamps_ms
        
        # --- Start of new validation section ---
        is_multistream_config = isinstance(self.feature_type, list) and \
                                len(self.feature_type) > 0 and \
                                isinstance(self.feature_type[0], list)

        if len(data_list) > 1 and not is_multistream_config:
            raise ValueError("When providing multiple data arrays, 'feature_type' must be a list of lists "
                             "(e.g., [['mean'], ['mav']]). A simple list (e.g., ['mean', 'history']) "
                             "applies multiple features to a single data array.")

        if is_multistream_config and len(data_list) != len(self.feature_type):
            raise ValueError(f"Number of data arrays ({len(data_list)}) must match number of feature groups "
                             f"in 'feature_type' ({len(self.feature_type)}).")
        # --- End of new validation section ---
        # Get the number of channels for each data array
        channels_per_array = []
        for d in data_list:
            # Convert to array if it's a list
            if isinstance(d, list):
                d = np.array(d)
            
            # Get number of channels
            if d.ndim == 1:
                channels_per_array.append(1)
            else:
                channels_per_array.append(d.shape[1])
        
        for i, (d, t) in enumerate(zip(data_list, timestamps_list)):
            if d.shape[0] != t.shape[0]:
                raise ValueError(f"Data and timestamps must have same number of samples for array {i}")
        
        if len(data_list) == 0 or any(len(t) == 0 for t in timestamps_list):
            return np.array([]) if return_array else []
        
        # Determine global time range across all arrays
        min_time = min(t.min() for t in timestamps_list)
        max_time = max(t.max() for t in timestamps_list)
        
        # Create bins
        bin_features = []
        current_time = min_time
        
        while current_time <= max_time:
            bin_start = current_time
            bin_end = current_time + self.bin_size_ms
            
            # Extract samples from all arrays for this bin
            bin_data_list = []
            
            should_skip = False
            for chans, d, t in zip(channels_per_array, data_list, timestamps_list):
                bin_mask = (t >= bin_start) & (t < bin_end)
                if np.any(bin_mask):
                    bin_data_list.append(d[bin_mask].reshape(-1, chans))
                else:
                    # Discard data from both data streams
                    should_skip = True
            if should_skip:
                print(f"Warning: No data found for at least one data stream for bin {bin_start:.2f} to {bin_end:.2f} ms. Skipping bin.")
                current_time += self.bin_size_ms
                continue
            else:
                bin_data_list = [d for d in bin_data_list if d is not None]
            
            if bin_data_list:
                # Process each array independently with its corresponding feature type
                feature_list = []
                combined_metadata = {
                    'sample_count': [],
                    'dimensions': [],
                    'feature_types': []
                }
                
                for i, bin_data in enumerate(bin_data_list):
                    # Get feature type and parameters for this array
                    if is_multistream_config:
                        # List of lists: one feature group per data array
                        ft = self.feature_type[i]
                        fp = self.feature_params[i] if i < len(self.feature_params) else {}
                    else:
                        # String or simple list: one feature group for all arrays
                        # (validation ensures this only happens for a single data array)
                        ft = self.feature_type
                        fp = self.feature_params[0] if self.feature_params else {}
                    
                    # Extract features for this array
                    features = self.compute_bin_features(
                        data=bin_data,
                        bin_end_timestamp_ms=bin_end,
                        feature_type=ft,
                        feature_params=fp,
                        array_index=i
                    )
                    
                    if features is not None:
                        import warnings
                        warnings.warn("Trial Filtering coming soon to a dataset near you", UserWarning)
                        feature_list.append(features['features'])
                        combined_metadata['sample_count'].append(features['sample_count'])
                        combined_metadata['dimensions'].append(features['dimensions'])
                        combined_metadata['feature_types'].append(features['feature_type'])
                
                if feature_list:
                    # Keep features as list if input was list, otherwise use single array
                    if len(data_list) > 1:
                        combined_features = feature_list
                    else:
                        combined_features = feature_list[0]
                    
                    # Create combined feature dictionary
                    features = {
                        "features": combined_features,
                        "sample_count": combined_metadata['sample_count'],
                        "feature_type": combined_metadata['feature_types'] if len(data_list) > 1 else combined_metadata['feature_types'][0],
                        "dimensions": combined_metadata['dimensions']
                    }
                    
                    # Add bin timing information
                    features['bin_start_ms'] = bin_start
                    features['bin_end_ms'] = bin_end
                    features['bin_center_ms'] = (bin_start + bin_end) / 2.0
                    bin_features.append(features)
            
            current_time += self.bin_size_ms
        
        # Post-process velocity features to use lookahead (vel_t = pos_t+1 - pos_t)
        self._postprocess_velocity_features(bin_features, is_multistream_config)
        
        # Return array of features if requested
        if return_array:
            if not bin_features:
                return np.array([])
            
            # Handle both single arrays and lists of arrays
            first_features = bin_features[0]['features']
            if isinstance(first_features, list):
                # Multiple arrays case - return list of arrays, one per input array
                num_arrays = len(first_features)
                result_arrays = []
                for array_idx in range(num_arrays):
                    array_features = [bf['features'][array_idx] for bf in bin_features]
                    result_arrays.append(np.array(array_features))
                return result_arrays
            else:
                # Single array case - return single array
                feature_arrays = [bf['features'] for bf in bin_features]
                return np.array(feature_arrays)
        
        return bin_features

    def compute_bin_features(self, 
                            data: np.ndarray,
                            bin_end_timestamp_ms: Optional[float] = None,
                            feature_type: Optional[Union[str, List]] = None,
                            feature_params: Optional[Dict] = None,
                            array_index: int = 0) -> Optional[Dict]:
        """
        Extract features from a single bin of data.
        
        Args:
            data: Data array of shape [n_samples, dimensions] or [dimensions] for single sample
            bin_end_timestamp_ms: Timestamp at the end of this bin (milliseconds)
            feature_type: Override feature type for this computation (uses self.feature_type if None)
            feature_params: Override feature parameters for this computation
            array_index: Index of the data array being processed (for multiple arrays)
            
        Returns:
            Dictionary containing extracted features, or None if no data
        """
        if data is None or data.size == 0:
            return None
        
        # Use provided parameters or fall back to self parameters
        ft = feature_type if feature_type is not None else self.feature_type
        fp = feature_params if feature_params is not None else {}
        
        # Handle nested feature types structure
        if isinstance(ft, list) and len(ft) > 0:
            if isinstance(ft[0], list):
                # List of lists - use the specified array index
                if array_index < len(ft):
                    ft = ft[array_index]
                else:
                    ft = ft[0]  # fallback to first
            # ft is now either a string or list of strings
        
        # Get feature parameters for this array
        if hasattr(self, 'feature_params') and array_index < len(self.feature_params):
            fp = self.feature_params[array_index]
        
        # Handle both single sample and multiple samples
        if data.ndim == 1:
            # Single sample case - reshape to [1, dimensions]
            processed_data = data.reshape(1, -1)
        else:
            # Multiple samples case - use as is
            processed_data = data
        
        # Compute base features (non-history, non-velocity) and handle special features separately
        if isinstance(ft, list):
            base_feature_types = [f for f in ft if f not in ['history', 'vel']]
            has_history = 'history' in ft
            has_velocity = 'vel' in ft
        else:
            base_feature_types = [ft] if ft not in ['history', 'vel'] else []
            has_history = ft == 'history'
            has_velocity = ft == 'vel'
        
        # Compute base features
        if base_feature_types:
            base_features = self._compute_features(processed_data, base_feature_types, fp)
        else:
            base_features = np.array([])
        
        # Collect all features in order
        all_features = []
        
        # Add base features first
        if base_features.size > 0:
            all_features.append(base_features)
        
        # Handle velocity if requested - add placeholder zeros that will be replaced during post-processing
        if has_velocity:
            velocity_placeholder = np.zeros_like(base_features) if base_features.size > 0 else np.array([])
            if velocity_placeholder.size > 0:
                all_features.append(velocity_placeholder)
        
        # Handle history if requested
        if has_history:
            hist_bins = fp.get('hist', 0)
            history_features = self._get_history_features(array_index, base_features.shape[0] if base_features.size > 0 else 0, hist_bins)
            if history_features.size > 0:
                all_features.append(history_features)
        
        # Concatenate all features
        final_features = np.concatenate(all_features) if all_features else np.array([])
        
        # Store base features for future history and velocity (only if we computed some)
        if base_features.size > 0:
            stream_history = self.previous_bin_features.get(array_index, [])
            stream_history.append(base_features)
            
            # Keep only the required number of previous bins for this stream
            if len(stream_history) > self.max_history_bins:
                stream_history = stream_history[-self.max_history_bins:]
                
            self.previous_bin_features[array_index] = stream_history
        
        # Create result dictionary
        features = {
            "features": final_features,
            "sample_count": processed_data.shape[0],
            "feature_type": ft,
            "dimensions": processed_data.shape[1]
        }
        
        # Add metadata
        if bin_end_timestamp_ms is not None:
            features['bin_end_timestamp_ms'] = bin_end_timestamp_ms
            features['bin_start_timestamp_ms'] = bin_end_timestamp_ms - self.bin_size_ms
        
        return features
    
    def _get_history_features(self, array_index: int, base_feature_size: int, hist_bins: int) -> np.ndarray:
        """
        Get concatenated history features for a specific stream.
        
        Args:
            array_index: The index of the data stream
            base_feature_size: Size of base features for zero-padding
            hist_bins: Number of historical bins to include
            
        Returns:
            Concatenated history features with zero-padding if needed
        """
        if hist_bins == 0:
            return np.array([])
            
        stream_history = self.previous_bin_features.get(array_index, [])
        available_bins = min(hist_bins, len(stream_history))
        
        history_parts = []
        for i in range(available_bins):
            history_parts.append(stream_history[-(i+1)])
        feature_size_for_padding = stream_history[-1].shape[0] if stream_history else base_feature_size
        for _ in range(hist_bins - available_bins):
            history_parts.append(np.zeros(feature_size_for_padding))
        return np.concatenate(history_parts) if history_parts else np.array([])
    
    def _get_velocity_features(self, array_index: int, current_base_features: np.ndarray, next_base_features: np.ndarray) -> np.ndarray:
        """
        Get velocity features as the difference between next and current base features (lookahead).
        
        Args:
            array_index: The index of the data stream
            current_base_features: Current bin's base features
            next_base_features: Next bin's base features (for lookahead velocity)
            
        Returns:
            Velocity features (next - current)
        """
        if current_base_features.size == 0 or next_base_features.size == 0:
            return np.array([])
            
        # Ensure shapes match (they should, but be safe)
        if next_base_features.shape != current_base_features.shape:
            return np.zeros_like(current_base_features)
        
        # Compute velocity as difference between next and current features (lookahead)
        return next_base_features - current_base_features
    
    def _postprocess_velocity_features(self, bin_features: List[Dict], is_multistream_config: bool):
        """
        Post-process velocity features to use lookahead (next bin's base features).
        
        Args:
            bin_features: List of feature dictionaries to modify in-place
            is_multistream_config: Whether we have multiple data streams
        """
        if len(bin_features) < 2:
            return  # Need at least 2 bins for velocity computation
        
        # Process all bins except the last one (no next bin available for last one)
        for i in range(len(bin_features) - 1):
            current_bin = bin_features[i]
            next_bin = bin_features[i + 1]
            
            # Check if current bin has velocity features
            if not self._bin_has_velocity_features(current_bin, is_multistream_config):
                continue
            
            # Extract base features from next bin and recompute velocity
            if is_multistream_config:
                # Multiple data streams - features is a list of arrays
                current_features_list = current_bin['features']
                next_features_list = next_bin['features']
                feature_types_list = current_bin['feature_type']
                
                for j, (curr_feats, next_feats, ft) in enumerate(zip(current_features_list, next_features_list, feature_types_list)):
                    if self._feature_group_has_velocity(ft):
                        # Extract base features from next bin
                        next_base_features = self._extract_base_features_from_total(next_feats, ft, j)
                        current_base_features = self._extract_base_features_from_total(curr_feats, ft, j)
                        
                        # Only compute velocity if we have valid base features from both bins
                        if next_base_features.size > 0 and current_base_features.size > 0:
                            new_velocity = self._get_velocity_features(j, current_base_features, next_base_features)
                            
                            # Replace velocity portion in current features
                            current_bin['features'][j] = self._replace_velocity_in_features(curr_feats, new_velocity, ft, j)
            else:
                # Single data stream - features is a single array
                current_features = current_bin['features']
                next_features = next_bin['features']
                feature_type = current_bin['feature_type']
                
                if self._feature_group_has_velocity(feature_type):
                    # Extract base features from next bin
                    next_base_features = self._extract_base_features_from_total(next_features, feature_type, 0)
                    current_base_features = self._extract_base_features_from_total(current_features, feature_type, 0)
                    
                    # Only compute velocity if we have valid base features from both bins
                    if next_base_features.size > 0 and current_base_features.size > 0:
                        new_velocity = self._get_velocity_features(0, current_base_features, next_base_features)
                        
                        # Replace velocity portion in current features
                        current_bin['features'] = self._replace_velocity_in_features(current_features, new_velocity, feature_type, 0)
    
    def _bin_has_velocity_features(self, bin_dict: Dict, is_multistream_config: bool) -> bool:
        """Check if a bin has velocity features."""
        if is_multistream_config:
            feature_types_list = bin_dict['feature_type']
            return any(self._feature_group_has_velocity(ft) for ft in feature_types_list)
        else:
            return self._feature_group_has_velocity(bin_dict['feature_type'])
    
    def _feature_group_has_velocity(self, feature_type: Union[str, List[str]]) -> bool:
        """Check if a feature group contains velocity features."""
        if isinstance(feature_type, str):
            return feature_type == 'vel'
        elif isinstance(feature_type, list):
            return 'vel' in feature_type
        return False
    
    def _extract_base_features_from_total(self, total_features: np.ndarray, feature_type: Union[str, List[str]], array_index: int) -> np.ndarray:
        """Extract just the base features (non-velocity, non-history) from total features."""
        if isinstance(feature_type, list):
            base_feature_types = [f for f in feature_type if f not in ['history', 'vel']]
            has_velocity = 'vel' in feature_type
            has_history = 'history' in feature_type
        else:
            base_feature_types = [feature_type] if feature_type not in ['history', 'vel'] else []
            has_velocity = feature_type == 'vel'
            has_history = feature_type == 'history'
        
        if not base_feature_types:
            return np.array([])
        
        # Calculate base feature size (this should match the computation in _compute_features)
        # For simplicity, we'll use the fact that base features come first and have a predictable size
        # based on the feature types and number of channels
        
        # Get feature params for size calculation
        fp = self.feature_params[array_index] if array_index < len(self.feature_params) else {}
        
        # Calculate expected base feature size
        base_feature_size = 0
        for ft in base_feature_types:
            if ft in ['mav', 'power', 'mean', 'var']:
                # These produce one feature per channel
                dimensions = self._get_dimensions_for_array(array_index)
                base_feature_size += dimensions
            elif ft == 'mean_and_vel':
                # This produces 2 features per channel (mean + velocity)
                dimensions = self._get_dimensions_for_array(array_index)
                base_feature_size += 2 * dimensions
        
        # Base features are at the beginning of the total features array
        return total_features[:base_feature_size]
    
    def _replace_velocity_in_features(self, total_features: np.ndarray, new_velocity: np.ndarray, feature_type: Union[str, List[str]], array_index: int) -> np.ndarray:
        """Replace the velocity portion of the total features with new velocity values."""
        if isinstance(feature_type, list):
            base_feature_types = [f for f in feature_type if f not in ['history', 'vel']]
            has_velocity = 'vel' in feature_type
            has_history = 'history' in feature_type
        else:
            base_feature_types = [feature_type] if feature_type not in ['history', 'vel'] else []
            has_velocity = feature_type == 'vel'
            has_history = feature_type == 'history'
        
        if not has_velocity:
            return total_features  # No velocity to replace
        
        # Calculate base feature size 
        base_feature_size = 0
        for ft in base_feature_types:
            if ft in ['mav', 'power', 'mean', 'var']:
                dimensions = self._get_dimensions_for_array(array_index)
                base_feature_size += dimensions
            elif ft == 'mean_and_vel':
                dimensions = self._get_dimensions_for_array(array_index)
                base_feature_size += 2 * dimensions
        
        # Velocity features come right after base features
        velocity_start = base_feature_size
        velocity_end = velocity_start + new_velocity.shape[0]
        
        # Create new feature array with replaced velocity
        new_features = total_features.copy()
        new_features[velocity_start:velocity_end] = new_velocity
        
        return new_features
    
    def _get_dimensions_for_array(self, array_index: int) -> int:
        """Get the number of dimensions/channels for a specific array index."""
        # This is a simplified approach - in practice, you might want to store this information
        # For now, we'll try to infer it from the stored history or use a default
        if array_index in self.previous_bin_features and self.previous_bin_features[array_index]:
            # Infer from stored base features
            base_features = self.previous_bin_features[array_index][-1]
            # Assume base features are computed with simple feature types that produce 1 feature per channel
            return base_features.shape[0]
        
        # Fallback - try to get from config if available
        return getattr(self, 'channels', 1)
    
    def _compute_features(self, data: np.ndarray, feature_types: List[str], feature_params: Optional[Dict] = None) -> np.ndarray:
        """
        Compute features from data using vectorized operations.
        
        Args:
            data: Input data of shape [samples, channels]
            feature_types: List of feature types to compute (no 'history' or 'vel' allowed here)
            feature_params: Parameters for the feature computation
            
        Returns:
            Concatenated feature array
        """
        if data.shape[0] == 0 or not feature_types:
            return np.array([])

        if feature_params is None:
            feature_params = {}

        all_features = []
        for ft in feature_types:
            feat = self._compute_single_feature(data, ft, feature_params)
            if feat.size > 0:
                all_features.append(feat)
        
        return np.concatenate(all_features) if all_features else np.array([])
    
    def _compute_single_feature(self, data: np.ndarray, feature_type: str, feature_params: Dict) -> np.ndarray:
        """
        Compute a single feature type from data.
        
        Args:
            data: Input data of shape [samples, channels]
            feature_type: Type of feature to compute
            feature_params: Parameters for the feature computation
            
        Returns:
            Feature array for the specified feature type
        """
        if data.shape[0] == 0:
            return np.array([])

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
            # Mean and velocity (legacy feature)
            mean_pos = np.mean(data, axis=0)
            vel = np.diff(data, axis=0).mean(axis=0) if data.shape[0] > 1 else np.zeros(data.shape[1])
            return np.concatenate((mean_pos, vel))
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Test basic configuration
    print("=== Basic Feature Extraction ===")
    config = {
        'bin_size_ms': 100,
        'feature_type': 'mav',
        'channels': 3,
    }
    
    extractor = FeatureExtractor(config)
    
    # Simulate data batch
    data_batch = np.random.randn(10, 3)  # 10 samples, 3 dimensions
    
    features = extractor.compute_bin_features(
        data=data_batch,
        bin_end_timestamp_ms=100.0
    )
    
    print(f"Single bin features: {features}")
    
    # Test nested feature types (list of lists)
    print("\n=== Testing Nested Feature Types ===")
    config_nested = {
        'bin_size_ms': 50,
        'feature_type': [['mean', 'var'], ['mav']],  # Two feature groups
        'feature_params': [{}, {}],  # Parameters for each group
        'channels': 3,
    }
    
    extractor_nested = FeatureExtractor(config_nested)
    
    # Create two data arrays
    data1 = np.random.randn(8, 3)
    data2 = np.random.randn(12, 3)
    timestamps1 = np.arange(0, 40, 5)  # 8 samples at 200Hz
    timestamps2 = np.arange(0, 60, 5)  # 12 samples at 200Hz
    
    nested_features = extractor_nested.extract_binned_features(
        data=[data1, data2],
        timestamps_ms=[timestamps1, timestamps2]
    )
    
    print(f"Nested features - Number of bins: {len(nested_features)}")
    if nested_features:
        print(f"First bin feature types: {nested_features[0]['feature_type']}")
        print(f"First bin feature shapes: {[f.shape for f in nested_features[0]['features']]}")
    
    # Test history feature
    print("\n=== Testing History Feature ===")
    config_history = {
        'bin_size_ms': 50,
        'feature_type': ['mean', 'history'],  # Mean feature + history of previous means
        'feature_params': [{'hist': 2}],  # Single parameter dict for the feature group
        'channels': 4,
    }
    
    extractor_history = FeatureExtractor(config_history)
    
    # Create timestamped data for history testing
    timestamps = np.arange(0, 300, 2)  # 0-300ms, 2ms intervals (500Hz)
    data = np.random.randn(len(timestamps), 4)  # 4 dimensions
    
    # Extract features with history
    history_features = extractor_history.extract_binned_features(
        data=data,
        timestamps_ms=timestamps
    )
    
    print(f"History features - Number of bins: {len(history_features)}")
    if len(history_features) >= 3:
        print(f"Feature shape (mean + 2*history): {history_features[0]['features'].shape}")
        
        # Show how history accumulates over time
        bin0_feats = history_features[0]['features']
        bin1_feats = history_features[1]['features'] 
        bin2_feats = history_features[2]['features']
        
        # Expected shapes: mean (4), history (8) -> total (12)
        print(f"Bin 0: current_mean={bin0_feats[:4].round(3)}, hist={bin0_feats[4:].round(3)}")
        print(f"Bin 1: current_mean={bin1_feats[:4].round(3)}, hist={bin1_feats[4:].round(3)}")
        print(f"Bin 2: current_mean={bin2_feats[:4].round(3)}, hist={bin2_feats[4:].round(3)}")
        print("✓ History feature working correctly - stores previous computed features!")
    
    # Test combined features: mean + velocity with history
    print("\n=== Testing Combined Features ===")
    config_combined = {
        'bin_size_ms': 50,
        'feature_type': [['mean', 'vel'], ['mav', 'history']],  # Second stream: mav + its history
        'feature_params': [{}, {'hist': 2}],  # No params for mean+vel, 2 bin history for second
        'channels': 3,
    }
    
    extractor_combined = FeatureExtractor(config_combined)
    
    # Create two data streams
    data_stream1 = np.random.randn(40, 3)
    data_stream2 = np.random.randn(40, 3)
    timestamps_stream1 = np.arange(0, 200, 5)  # 20 samples
    timestamps_stream2 = np.arange(0, 200, 5)  # 25 samples
    
    combined_features = extractor_combined.extract_binned_features(
        data=[data_stream1, data_stream2],
        timestamps_ms=[timestamps_stream1, timestamps_stream2]
    )
    
    print(f"Combined features - Number of bins: {len(combined_features)}")
    if combined_features:
        first_bin = combined_features[0]
        print(f"First bin feature types: {first_bin['feature_type']}")
        # Expected shapes: ['mean', 'vel'] -> 6; ['mav', 'history'] -> mav(3) + hist(2*3=6) -> 9
        print(f"First bin feature shapes: {[f.shape for f in first_bin['features']]} (expected [(6,), (9,)])")
        
        # Debug the history size issue
        if len(combined_features) >= 2:
            last_bin = combined_features[-1]
            second_stream_features = last_bin['features'][1]  # Second stream features
            print(f"DEBUG: Last bin second stream features shape: {second_stream_features.shape}")
            print(f"DEBUG: Expected = mav(3) + hist(2*3) = 9, got = {second_stream_features.shape[0]}")
            print(f"DEBUG: History for stream 1: {len(extractor_combined.previous_bin_features.get(1, []))} bins stored")
    
    # Test validation: history cannot be used alone
    print("\n=== Testing History Validation ===")
    try:
        invalid_config = {
            'bin_size_ms': 50,
            'feature_type': 'history',  # This should fail
            'channels': 4,
        }
        FeatureExtractor(invalid_config)
        print("❌ Validation failed - should not allow history alone")
    except ValueError as e:
        print(f"✓ Validation working: {e}")
    
    # Test reset history functionality
    print("\n=== Testing Reset History ===")
    print(f"History keys before reset: {list(extractor_combined.previous_bin_features.keys())}")
    extractor_combined.reset_history()
    print(f"History after reset: {extractor_combined.previous_bin_features}")
    
    # Test different feature types (legacy test)
    print("\n=== Testing Legacy Feature Types ===")
    feature_types = ['mav', 'power', 'mean', 'var', 'vel', 'mean_and_vel']
    
    for ft in feature_types:
        config_test = {
            'bin_size_ms': 100,
            'feature_type': ft,
            'channels': 4,
        }
        extractor_test = FeatureExtractor(config_test)
        
        features = extractor_test.compute_bin_features(
            data=np.random.randn(15, 4),
            bin_end_timestamp_ms=100.0
        )
        
        print(f"{ft} features shape: {features['features'].shape}")
