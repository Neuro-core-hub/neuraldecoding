import neuraldecoding.utils
import neuraldecoding.stabilization.latent_space_alignment
import neuraldecoding.dataaugmentation.DataAugmentation
from neuraldecoding.dataaugmentation import SequenceScaler
from neuraldecoding.feature_extraction import FeatureExtractor
from neuraldecoding.utils.utils_general import resolve_path
import sklearn.preprocessing

import torch

from abc import ABC, abstractmethod

import time
import pickle
from typing import Union, List
import numpy as np

class PreprocessingBlock(ABC):
	"""
	Base class for preprocessing blocks in the pipeline.
	Each block should implement the `transform` method to process the data.
	"""
	def __init__(self):
		"""
		Initializes the PreprocessingBlock.
		"""
		pass

	@abstractmethod
	def transform(self, data, interpipe):
		"""
		Transform the data.

		Args:
			data (any): The input data to be transformed.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		"""
		pass

class DataFormattingBlock(PreprocessingBlock):
	"""
	Base class for data formatting blocks in the pipeline.
	Act as a reminder that the block is used to modify the data format.
	"""
	def __init__(self):
		"""
		Initializes the DataFormattingBlock.
		"""
		pass

	@abstractmethod
	def transform(self, data, interpipe):
		"""
		Transform the data.

		Args:
			data (any): The input data to be transformed.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		"""
		pass

class DataProcessingBlock(PreprocessingBlock):
	"""
	Base class for data processing blocks in the pipeline.
	Act as a reminder that the block is used to modify the data itself.
	"""
	def __init__(self):
		"""
		Initializes the DataProcessingBlock.
		"""
		pass

	@abstractmethod
	def transform(self, data, interpipe):
		"""
		Transform the data.

		Args:
			data (any): The input data to be transformed.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		"""
		pass


# Wrappers that Modifies Data Format
class Dict2DataDictBlock(DataFormattingBlock):
	"""
	Converts a dictionary (from load_one_nwb) to neural and finger data in dictionary format.
	Add 'trial_idx' to interpipe.
	"""
	def __init__(self, neural_type = "sbp"):
		"""
		Initializes the Dict2DataDictBlock.
		Args:
			neural_type (str): Type of neural data to extract from the dictionary. Default is "sbp". Can be "sbp" or "tcfr"
		"""
		super().__init__()
		self.neural_type = neural_type

	def transform(self, data: dict, interpipe):
		"""
		Transform the data from a dictionary to neural and finger data in dictionary format.
		Args:
			data (dict): Input data dictionary from `neuraldecoding.utils.data_tools.load_one_nwb()`.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing 'neural' and 'finger' data.
			interpipe (dict): Updated interpipe dictionary with entry of 'trial_idx' containing the trial indices.
		"""
		(neural, finger), trial_idx = neuraldecoding.utils.neural_finger_from_dict(data, self.neural_type)
		interpipe['trial_idx'] = trial_idx

		data_out = {'neural': neural, 'finger': finger}
		return data_out, interpipe

class ClassificationDict2TupleBlock(DataFormattingBlock):
	"""
	Converts a dictionary to a tuple format for classification tasks.
	Temporary block designed specifically for LDA and data from RPNI C-P2 experiments.

	Assumes the dictionary contains keys 'X_train', 'y_train', 'X_test', and 'y_test'.
	Returns a tuple of (neural, finger) for training or testing based on the 'is_train' key in the interpipe dictionary.
	"""
	def __init__(self):
		super().__init__()
	def transform(self, data: dict, interpipe):
		"""
		Transform the data from a dictionary to a tuple format for classification tasks.
		Args:
			data (dict): Input data dictionary containing 'X_train', 'y_train', 'X_test', and 'y_test'.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (tuple): A tuple containing either:
				- (neural, finger) , contains either training or testing data depending on 'is_train' in interpipe.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if 'is_train' not in interpipe:
			raise ValueError("ClassificationDict2TupleBlock requires 'is_train' in interpipe.")

		if interpipe['is_train']:
			neural, finger = (data['X_train'], data['y_train'].flatten())
		else:
			neural, finger = (data['X_test'], data['y_test'].flatten())
		data_out = (neural, finger)
		return data_out, interpipe

class DataSplitBlock(DataFormattingBlock):
	"""
	A block for splitting data into training and testing sets using sklearn's train_test_split.
	Splits multiple data arrays using the same train/test indices.
	"""
	def __init__(self, location: List[str] = ['neural', 'finger'], 
				 split_ratio: float = 0.8, split_seed: int = 42, shuffle: bool = True):
		"""
		Initializes the DataSplitBlock.
		Args:
			location (List[str]): List of keys for data arrays in the data dictionary. Default is ['neural', 'finger'].
			split_ratio (float): The ratio of training data to total data. Default is 0.8.
			split_seed (int): Seed for random number generator to ensure reproducibility. Default is 42.
			shuffle (bool): Whether to shuffle the data before splitting. Default is True.
		"""
		super().__init__()
		self.location = location
		self.split_ratio = split_ratio
		self.split_seed = split_seed
		self.shuffle = shuffle

	def transform(self, data, interpipe):
		"""
		Transform the data by splitting it into training and testing sets using sklearn's train_test_split.
		All data arrays are split using the same train/test indices.
		Args:
			data (dict): Input data dictionary containing the specified data keys.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing:
				- '{key}_train': Training data for each key
				- '{key}_test': Testing data for each key
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		from sklearn.model_selection import train_test_split
		
		# Validate that all required keys exist
		for key in self.location:
			if key not in data:
				raise ValueError(f"DataSplitBlock requires '{key}' key in data dictionary.")
		
		# Collect all data arrays
		data_arrays = [data[key] for key in self.location]
		
		# Split all arrays using the same indices
		split_results = train_test_split(
			*data_arrays,
			train_size=self.split_ratio,
			random_state=self.split_seed,
			shuffle=self.shuffle
		)
		
		# Organize results into train/test pairs
		data_out = {}
		for i, key in enumerate(self.location):
			train_idx = i * 2
			test_idx = i * 2 + 1
			data_out[f'{key}_train'] = split_results[train_idx]
			data_out[f'{key}_test'] = split_results[test_idx]
		
		return data_out, interpipe

class Dict2TupleBlock(DataFormattingBlock):
	"""
	Converts a dictionary to a tuple format.
	Accepts either 2 or 4 keys in the dictionary:
		- If 2 keys: 'neural' and 'finger'
		- If 4 keys: 'neural_train', 'neural_test', 'finger_train', 'finger_test'
	"""
	def __init__(self):
		super().__init__()

	def transform(self, data, interpipe):
		"""
		Transform the data from a dictionary to a tuple format.
		Args:
			data (dict): Input data dictionary.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (tuple): A tuple containing either:
				- (neural, finger) if 2 keys are present
				- (neural_train, neural_test, finger_train, finger_test) if 4 keys are present
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if len(data) == 2:
			data_out = (data['neural'] , data['finger'])
		elif len(data) == 4:
			data_out = (data['neural_train'], data['neural_test'], data['finger_train'], data['finger_test'])
		else:
			raise ValueError(f"Data Dict Contain Unexpected # of Keys. Expected 2 or 4 keys, got {len(data)}")
		return data_out, interpipe

class Dataset2DictBlock(DataFormattingBlock):
	"""
	Converts a dictionary (from load_one_nwb) to neural and finger data in dictionary format.
	Add 'trial_idx' to interpipe.
	"""
	def __init__(self, neural_nwb_loc, behavior_nwb_loc, apply_trial_filtering = True):
		"""
		Initializes the Dataset2DictBlock.
		Args:
			neural_nwb_loc (str): Location path for neural data in the NWB file.
			behavior_nwb_loc (str): Location path for behavior data in the NWB file.
			apply_trial_filtering (bool): Whether to apply trial filtering. Default is True.
		"""
		self.neural_nwb_loc = neural_nwb_loc
		self.behavior_nwb_loc = behavior_nwb_loc
		self.apply_trial_filtering = apply_trial_filtering
		super().__init__()

	def transform(self, data, interpipe):
		"""
		Transform the data from a dictionary to neural and finger data in dictionary format.
		Args:
			data (dict): Input dataset class.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing 'neural' and 'finger' data.
			interpipe (dict): Updated interpipe dictionary with entry of 'trial_idx' containing the trial indices.
		"""
		#TODO: Implement trial filtering (have an apply trial filters feature)
		neural_nwb, behavior_nwb = resolve_path(data.dataset, self.neural_nwb_loc), resolve_path(data.dataset, self.behavior_nwb_loc)
		neural, behavior = neural_nwb.data[:], behavior_nwb.data[:]
		# Convert timestamps to milliseconds
		neural_ts, behavior_ts = neural_nwb.timestamps[:] * 1000, behavior_nwb.timestamps[:] * 1000
		if self.apply_trial_filtering:
			UserWarning("Trial Filtering coming soon to a dataset near you")

		data_out = {'neural': neural, 'neural_ts': neural_ts, 'behavior': behavior, 'behavior_ts': behavior_ts}
		return data_out, interpipe

class IndexSelectorBlock(DataFormattingBlock):
	"""
	A block for selecting data from a dictionary based on indices.
	"""
	def __init__(self, location: list[str], indices: Union[int, list]):
		super().__init__()
		self.location = location
		self.indices = indices

	def transform(self, data, interpipe):
		data_out = data.copy()
		for loc in self.location:
			if loc not in data:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
			# Convert data to numpy array if not already
			if not isinstance(data[loc], np.ndarray):
				data[loc] = np.array(data[loc])
			if data[loc].ndim == 1:
				# For 1D data, select indices directly
				data_out[loc] = data[loc][self.indices]
			elif data[loc].ndim == 2:
				# For 2D data, select indices from the second dimension
				data_out[loc] = data[loc][:, self.indices]
			else:
				raise ValueError(f"Data at location '{loc}' must be 1D or 2D, got {data[loc].ndim}D")
		return data_out, interpipe

class OneHotToClassNumberBlock(DataFormattingBlock):
	"""
	A block for converting one-hot encoded data to class numbers.
	"""
	def __init__(self, location):
		super().__init__()
		self.location = location
	def transform(self, data, interpipe):
		data_out = data.copy()
		for loc in self.location:
			if loc not in data:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
			data_out[loc] = np.argmax(data[loc], axis=1)
		return data_out, interpipe

class RoundToIntegerBlock(DataFormattingBlock):
	"""
	A block for rounding data to integers.
	"""
	def __init__(self, location):
		super().__init__()
		self.location = location
	def transform(self, data, interpipe):
		data_out = data.copy()
		for loc in self.location:
			if loc not in data:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
			data_out[loc] = np.round(data[loc])
		return data_out, interpipe

# Wrappers that Modify Data
class StabilizationBlock(DataProcessingBlock):
	"""
	A block for stabilizing the latent space of neural data.
	Have different behavior during training and testing phases, specified by the 'is_train' key in the interpipe dictionary.
	It uses a specified stabilization method from `neuraldecoding.stabilization.latent_space_alignment`.
	"""
	def __init__(self, location, stabilization_config):
		"""
		Initializes the StabilizationBlock.
		Args:
			location (str): The key in the data dictionary where the stabilization is applied.
			stabilization_config (dict): Configuration for the stabilization method.
		"""
		super().__init__()
		stabilization_method = getattr(neuraldecoding.stabilization.latent_space_alignment, stabilization_config["type"])
		self.stabilization = stabilization_method(stabilization_config["params"])
		self.location = location

	def transform(self, data, interpipe):
		"""
		Transform the data by stabilizing the latent space. calls stabilization.fit() during training and stabilization.extract_latent_space() during testing.
		Training and testing behavior is determined by the 'is_train' key in the interpipe dictionary.
		Args:
			data (dict): Input data dictionary containing the data to be stabilized.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data (dict): The data dictionary with the stabilized data at the specified location.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if 'is_train' not in interpipe:
			raise ValueError("The 'interpipe' dictionary for StabilizationBlock must contain an 'is_train' key.")

		if interpipe['is_train']:
			data[self.location] = self.stabilization.fit(data[self.location])
			self.stabilization.save_alignment()
		else:
			self.stabilization.load_alignment()
			data[self.location] = self.stabilization.extract_latent_space(data[self.location])
		return data, interpipe

class AddHistoryBlock(DataProcessingBlock):
	"""
	A block for adding history to the data at specified locations.
	It uses `neuraldecoding.utils.add_history_numpy` to add history.
	"""
	def __init__(self, location, seq_length = 10):
		"""
		Initializes the AddHistoryBlock.
		Args:
			location (str or list): The key(s) in the data dictionary where history is added.
			seq_length (int): The length of the history to be added. Default is 10.
		"""
		super().__init__()
		self.location = location
		self.seq_length = seq_length

	def transform(self, data, interpipe):
		"""
		Transform the data by adding history to the specified locations of datastream.
		Args:
			data (dict): Input data dictionary containing the data to which history is added.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data (dict): The data dictionary with history added at the specified locations.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if isinstance(self.location, str):
			self.location = [self.location]

		for loc in self.location:
			data[loc] = neuraldecoding.utils.add_history_numpy(data[loc], self.seq_length)

		return data, interpipe

class NormalizationBlock(DataProcessingBlock):
	"""
	A block for normalizing data using specified methods.
	Supports multiple normalization methods defined in `neuraldecoding.dataaugmentation.DataAugmentation`.
	"""
	def __init__(self, location, method, normalizer_params):
		"""
		Initializes the NormalizationBlock.
		Args:
			location (str or list): The key(s) in the data dictionary where normalization is applied.
			method (str): The normalization method to use. Options includes 'moving_average', 'sklearn', 'sequence_scaler'.
			normalizer_params (dict): Parameters for the normalization method.
		"""
		super().__init__()
		self.location = location
		self.normalizer_method = method
		self.normalizer_params = normalizer_params

	def transform(self, data, interpipe):
		"""
		Transform the data by normalizing it using the specified method.
		Args:
			data (dict): Input data dictionary containing the data to be normalized.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:	
			data (dict): The data dictionary with normalized data at the specified location.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if self.normalizer_method == 'moving_average':
			p = self.normalizer_params['params']
		elif self.normalizer_method == 'sklearn':
			normalizer = getattr(sklearn.preprocessing, self.normalizer_params['type'])
			p = {'normalizer': normalizer(**self.normalizer_params['params'])}
		else:
			p = {}
		for loc in self.location:
			data[loc], normalizer = neuraldecoding.dataaugmentation.DataAugmentation.normalize(data[loc],
																				   method = self.normalizer_method,
																				   **p)
		return data, interpipe

class UpdateNormalizationBlock(DataProcessingBlock):
	def __init__(self, location, method, normalizer_params):
		super().__init__()
		self.location = location
		self.normalizer_method = method
		self.normalizer_params = normalizer_params
	def transform(self, data, interpipe):
		if self.normalizer_method == 'sklearn':
			normalizer = getattr(sklearn.preprocessing, self.normalizer_params['type'])(**self.normalizer_params['params'])
			data[self.location[0]] = normalizer.fit_transform(data[self.location[0]])
			data[self.location[1]] = normalizer.transform(data[self.location[1]])
		elif self.normalizer_method == 'sequence_scaler':
			normalizer = SequenceScaler()
			data[self.location[0]] = normalizer.fit_transform(data[self.location[0]])
			data[self.location[1]] = normalizer.transform(data[self.location[1]])
		else:
			raise ValueError(f"Unsupported normalization method: {self.normalizer_method}")
		if self.normalizer_params['is_save']:
			if 'save_path' not in self.normalizer_params:
				raise ValueError("NormalizationBlock requires 'save_path' in normalizer_params when is_save is True.")
			with open(self.normalizer_params['save_path'], 'wb') as f:
				pickle.dump(normalizer, f)
		return data, interpipe

class LoadNormalizationBlock(DataProcessingBlock):
	def __init__(self, location, method, normalizer_params):
		super().__init__()
		self.location = location
		self.normalizer_method = method
		self.normalizer_params = normalizer_params
	def transform(self, data, interpipe):
		with open(self.normalizer_params['save_path'], 'rb') as f:
			normalizer = pickle.load(f)
		for loc in self.location:
			data[loc] = normalizer.transform(data[loc])
		return data, interpipe

class EnforceTensorBlock(DataProcessingBlock):
	"""
	A block for ensuring that all data in the dictionary is converted to PyTorch tensors with given device and dtype.
	"""
	def __init__(self, device='cpu', dtype=torch.float32):
		"""
		Initializes the EnforceTensorBlock.
		Args:
			device (str): The device to which the tensors should be moved. Default is 'cpu'.
			dtype (str): The data type of the tensors. Default is 'torch.float32'.
		"""
		super().__init__()
		self.device = device
		self.dtype = getattr(torch, dtype)

	def transform(self, data, interpipe):
		"""
		Transform the data by converting all elements in the dictionary to PyTorch tensors with specified device and dtype.
		Args:
			data (dict): Input data dictionary containing the data to be converted.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data (dict): The data dictionary with all elements converted to PyTorch tensors with specified device and dtype.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		for key in data:
			if isinstance(data[key], torch.Tensor):
				data[key] = data[key].to(self.device, dtype=self.dtype)
			else:
				data[key] = torch.tensor(data[key], device=self.device, dtype=self.dtype)
		return data, interpipe

class FeatureExtractionBlock(DataProcessingBlock):
	def __init__(self, location_data: list[str], location_ts: list[str], feature_extractor_config: dict):
		super().__init__()
		self.location_data = location_data
		self.location_ts = location_ts
		self.feature_extractor = FeatureExtractor(feature_extractor_config)

	def transform(self, data, interpipe):
		for loc in self.location_data + self.location_ts:
			if loc not in data:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
		features_list = self.feature_extractor.extract_binned_features(data=[data[loc] for loc in self.location_data], timestamps_ms=[data[loc] for loc in self.location_ts], return_array=True)
		# FIXME: for now, deleting the timestamps from the data dictionary
		# since they're not correct anymore
		for loc in self.location_ts:
			del data[loc]
		for i, loc in enumerate(self.location_data):
			data[loc] = features_list[i]
		return data, interpipe
	

class RegressionToClassificationBlock(DataProcessingBlock):
	"""
	A block for converting regression data into classification data by applying conditions to define classes.
	Each class is defined by a list of conditions (lambda functions) that must all be true for each column.
	Samples that don't match any defined conditions are automatically assigned to an additional "other" class.
	"""
	def __init__(self, location: str, conditions: List[List], output_key: str = None):
		"""
		Initializes the RegressionToClassificationBlock.
		Args:
			location (str): Key for the regression data array in the data dictionary.
			conditions (List[List]): List of class definitions. Each inner list contains conditions
									 for each column that define when a sample belongs to that class.
									 Conditions can be either lambda functions or strings.
									 String examples: "x < 0.2", "(x >= 0.2) & (x < 0.5)", "x > 0.8"
									 Lambda examples: lambda x: x < 0.2, lambda x: (x >= 0.2) & (x < 0.5)
									 Example: [
									 	["x < 0.2", "x < 0.2"],  # class 0
									 	["(x >= 0.2) & (x < 0.5)", "x < 0.2"],  # class 1
									 ]
									 Note: Samples not matching any conditions get assigned to an additional "other" class.
			output_key (str): Key for the output classification data. If None, uses same location (overwrites input).
		"""
		super().__init__()
		self.location = location
		self.conditions = self._convert_conditions(conditions)
		self.output_key = output_key if output_key is not None else location
		self.n_classes = len(conditions)
	
	def _convert_conditions(self, conditions):
		"""
		Convert string conditions to lambda functions if needed.
		Args:
			conditions: List of lists containing either strings or lambda functions
		Returns:
			List of lists containing lambda functions
		"""
		converted_conditions = []
		for class_conditions in conditions:
			converted_class = []
			for condition in class_conditions:
				if isinstance(condition, str):
					# Convert string to lambda function
					try:
						lambda_func = eval(f"lambda x: {condition}")
						converted_class.append(lambda_func)
					except Exception as e:
						raise ValueError(f"Invalid condition string '{condition}': {e}")
				else:
					# Assume it's already a callable (lambda function)
					converted_class.append(condition)
			converted_conditions.append(converted_class)
		return converted_conditions
	
	def transform(self, data, interpipe):
		"""
		Transform regression data into classification labels.
		Args:
			data (dict): Input data dictionary containing the regression data.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks.
		Returns:
			data_out (dict): Copy of input data with added classification labels.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if self.location not in data:
			raise ValueError(f"RegressionToClassificationBlock requires '{self.location}' key in data dictionary.")
		
		regression_data = data[self.location]
		n_samples, n_features = regression_data.shape
		
		# Validate conditions don't exceed number of features
		if self.conditions:
			n_conditions = len(self.conditions[0])
			if n_conditions > n_features:
				raise ValueError(f"Number of conditions per class ({n_conditions}) cannot exceed number of features ({n_features})")
			
			# Use only the first n_conditions features
			features_to_use = min(n_conditions, n_features)
		else:
			features_to_use = n_features
		
		# Initialize class labels (default to -1 for unclassified)
		class_labels = np.full(n_samples, -1, dtype=int)
		
		# Apply conditions for each class
		for class_idx, class_conditions in enumerate(self.conditions):
			# Create boolean mask for each feature condition (only use first features_to_use features)
			feature_masks = []
			for feature_idx, condition in enumerate(class_conditions[:features_to_use]):
				# Skip None conditions (no constraint on this feature)
				if condition is not None:
					feature_data = regression_data[:, feature_idx]
					feature_mask = condition(feature_data)
					feature_masks.append(feature_mask)
			
			# Combine all non-None feature conditions with AND logic
			# If no conditions were specified (all None), all samples match this class
			if feature_masks:
				class_mask = np.all(feature_masks, axis=0)
			else:
				class_mask = np.ones(n_samples, dtype=bool)  # All samples match if no conditions
			
			# Assign class label (prioritize earlier classes in case of overlap)
			unassigned_mask = class_labels == -1
			final_mask = class_mask & unassigned_mask
			class_labels[final_mask] = class_idx
		
		# Assign remaining unmatched samples to an "other" class
		unmatched_mask = class_labels == -1
		if np.any(unmatched_mask):
			other_class_idx = len(self.conditions)  # Next available class number
			class_labels[unmatched_mask] = other_class_idx
			self.n_classes = len(self.conditions) + 1
		else:
			self.n_classes = len(self.conditions)
		
		# Copy input data and add classification labels
		data_out = data.copy()
		data_out[self.output_key] = class_labels
		
		return data_out, interpipe
	
