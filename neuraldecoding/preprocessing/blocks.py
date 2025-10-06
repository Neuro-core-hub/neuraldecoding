import neuraldecoding.utils
import neuraldecoding.stabilization.latent_space_alignment
import neuraldecoding.dataaugmentation.DataAugmentation
from neuraldecoding.dataaugmentation import SequenceScaler
from neuraldecoding.feature_extraction import FeatureExtractor
from neuraldecoding.utils.utils_general import resolve_path
from neuraldecoding.utils.data_tools import load_one_nwb
from neuraldecoding.utils.training_utils import OutputScaler
from neuraldecoding.preprocessing.onset_detection import MovementOnsetDetector
import sklearn.preprocessing

import torch

from abc import ABC, abstractmethod

import warnings
import time
import pickle
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


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
	Converts a dictionary (from load_one_nwb) to neural and behaviour data in dictionary format.
	Add 'trial_idx' to interpipe.
	"""
	def __init__(self, neural_type = "sbp", data_keys = ["neural", "behaviour"], interpipe_keys = "trial_idx"):
		"""
		Initializes the Dict2DataDictBlock.
		Args:
			neural_type (str): Type of neural data to extract from the dictionary. Default is "sbp". Can be "sbp" or "tcfr"
			data_keys (list): List of 2 keys names to store the neural and behaviour data in the output dictionary. Default is ["neural", "behaviour"].
			interpipe_keys (str): Key name to store the trial_idx in the interpipe dictionary. Default is "trial_idx".
		"""
		super().__init__()
		self.neural_type = neural_type
		self.data_keys = data_keys
		self.interpipe_keys = interpipe_keys

	def transform(self, data: dict, interpipe):
		"""
		Transform the data from a dictionary to neural and behaviour data in dictionary format.
		Args:
			data (dict): Input data dictionary from `neuraldecoding.utils.data_tools.load_one_nwb()`.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing 'neural' and 'behaviour' data.
			interpipe (dict): Updated interpipe dictionary with entry of 'trial_idx' containing the trial indices.
		"""
		(neural, behaviour), trial_idx = neuraldecoding.utils.neural_finger_from_dict(data, self.neural_type)
		interpipe[self.interpipe_keys] = trial_idx

		data_out = {}
		if len(self.data_keys) >= 1:
			data_out[self.data_keys[0]] = neural
			data_out[self.data_keys[1]] = behaviour

		return data_out, interpipe
	
class LoadNWBBlock(DataFormattingBlock):
	'''
	Block for load NWB file in trainer testing, to bypass dataset to reduce potential problems (probably no but remove variabilities) from dataset.
	'''
	def __init__(self):
		super().__init__()

	def transform(self, data, interpipe):
		'''
		data is expected to be a dictionary with entry of data_path containing directory string to the NWB file
		'''
		data = load_one_nwb(data['data_path'])
		return data, interpipe
	
class ClassificationDict2TupleBlock(DataFormattingBlock):
	"""
	Converts a dictionary to a tuple format for classification tasks.
	Temporary block designed specifically for LDA and data from RPNI C-P2 experiments.

	Assumes the dictionary contains keys 'X_train', 'y_train', 'X_test', and 'y_test'.
	Returns a tuple of (neural, behaviour) for training or testing based on the 'is_train' key in the interpipe dictionary.
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
				- (neural, behaviour) , contains either training or testing data depending on 'is_train' in interpipe.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if 'is_train' not in interpipe:
			raise ValueError("ClassificationDict2TupleBlock requires 'is_train' in interpipe.")

		if interpipe['is_train']:
			neural, behaviour = (data['X_train'], data['y_train'].flatten())
		else:
			neural, behaviour = (data['X_test'], data['y_test'].flatten())
		data_out = (neural, behaviour)
		return data_out, interpipe

class DataKeyRenameBlock(DataFormattingBlock):
	"""
	A block for renaming keys in the data dictionary.
	"""
	def __init__(self, rename_map):
		"""
		Initializes the KeyRenameBlock.
		Args:
			rename_map (dict): A dictionary mapping old keys to new keys. Keys represent the old names, and values represent the new names.
		"""
		super().__init__()
		self.rename_map = rename_map

	def transform(self, data, interpipe):
		"""
		Renames keys in the data dictionary according to the rename_map.
		Args:
			data (dict): Input data dictionary.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data (dict): The data dictionary with renamed keys.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		for old_key, new_key in self.rename_map.items():
			if old_key in data:
				data[new_key] = data.pop(old_key)
			else:
				raise KeyError(f"Key '{old_key}' not found in data dictionary.")
		return data, interpipe

class InterpipeKeyRenameBlock(DataFormattingBlock):
	"""
	A block for renaming keys in the interpipe dictionary.
	"""
	def __init__(self, rename_map):
		"""
		Initializes the InterpipeKeyRenameBlock.
		Args:
			rename_map (dict): A dictionary mapping old keys to new keys. Keys represent the old names, and values represent the new names.
		"""
		super().__init__()
		self.rename_map = rename_map

	def transform(self, data, interpipe):
		"""
		Renames keys in the interpipe dictionary according to the rename_map.
		Args:
			data (dict): Input data dictionary.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data (dict): The data dictionary remains unchanged.
			interpipe (dict): The interpipe dictionary with renamed keys.
		"""
		for old_key, new_key in self.rename_map.items():
			if old_key in interpipe:
				interpipe[new_key] = interpipe.pop(old_key)
			else:
				raise KeyError(f"Key '{old_key}' not found in interpipe dictionary.")
		return data, interpipe

class DataSplitBlock(DataFormattingBlock):
	"""
	A block for splitting data into training and testing sets based on trial indices.
	Assumes the data dictionary contains 'neural' and 'behaviour' keys, and the interpipe dictionary contains 'trial_idx', if location and interpipe_location are not specified.
	It uses `neuraldecoding.utils.data_split_trial` to perform the split.
	"""
	def __init__(self, split_ratio: 0.8, split_seed: 42, location = ['neural', 'behaviour'], interpipe_location = ['trial_idx'], data_keys = ['neural_train', 'neural_test', 'behaviour_train', 'behaviour_test'], shuffle = False):
		"""
		Initializes the DataSplitBlock.
		Args:
			location (List[str]): List of keys for data arrays in the data dictionary. Default is ['neural', 'behaviour'].
			split_ratio (float): The ratio of training data to total data. Default is 0.8.
			split_seed (int): Seed for random number generator to ensure reproducibility. Default is 42.
			location (list): List of 2 keys names in the data dictionary to be split. Default is ['neural', 'behaviour'].
			interpipe_location (list): List of 1 key name in the interpipe dictionary containing the trial indices. Default is ['trial_idx'].
			data_keys (list): List of 4 keys names to store the split data in the output dictionary. Default is ['neural_train', 'neural_test', 'behaviour_train', 'behaviour_test'].
		"""
		super().__init__()
		self.location = location
		self.split_ratio = split_ratio
		self.split_seed = split_seed
		self.location = location
		self.interpipe_location = interpipe_location
		self.data_keys = data_keys
		self.shuffle = shuffle

	def transform(self, data, interpipe):
		"""
		Transform the data by splitting it into training and testing sets based on trial indices.
		Assumes the data dictionary contains 'neural' and 'behaviour' keys, and the interpipe dictionary contains 'trial_idx', if location and interpipe_location are not specified.
		It uses `neuraldecoding.utils.data_split_trial` to perform the split.
		Args:
			data (dict): Input data dictionary containing 'neural' and 'behaviour' keys if location are not specified.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing (by default data keys):
				- 'neural_train': Training neural data
				- 'neural_test': Testing neural data
				- 'behaviour_train': Training behaviour data
				- 'behaviour_test': Testing behaviour data
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		trial_idxs = interpipe.get(self.interpipe_location[0], None)
		if trial_idxs is None:
			print(f"Warning: DataSplitBlock requires {self.interpipe_location[0]} in interpipe from other wrappers (Dict2DataBlock).")

		split_data = neuraldecoding.utils.data_split_trial(data[self.location[0]], 
														   data[self.location[1]], 
														   trial_idx=trial_idxs, 
														   split_ratio=self.split_ratio, 
														   seed=self.split_seed,
														   shuffle=self.shuffle)

		(neural_train, behaviour_train), (neural_test, behaviour_test) = split_data
		data_out = {}
		data_out[self.data_keys[0]] = neural_train
		data_out[self.data_keys[1]] = neural_test
		data_out[self.data_keys[2]] = behaviour_train
		data_out[self.data_keys[3]] = behaviour_test
		return data_out, interpipe

class Dict2TupleBlock(DataFormattingBlock):
	"""
	Converts a dictionary to a tuple format.
	Accepts either 2 or 4 keys in the dictionary:
		- If 2 keys: 'neural' and 'behaviour', by default location (or no location).
		- If 4 keys: 'neural_train', 'neural_test', 'behaviour_train', 'behaviour_test', by default location (or no location).
	"""
	def __init__(self, location = None):
		super().__init__()
		self.location = location

	def transform(self, data, interpipe):
		"""
		Transform the data from a dictionary to a tuple format.
		Args:
			data (dict): Input data dictionary.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (tuple): A tuple containing either, (by default location):
				- (neural, behaviour) if 2 keys are present
				- (neural_train, neural_test, behaviour_train, behaviour_test) if 4 keys are present
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if self.location is None:
			if len(data) == 2:
				self.location = ['neural', 'behaviour']
			elif len(data) == 4:
				self.location = ['neural_train', 'neural_test', 'behaviour_train', 'behaviour_test']

		if len(data) == 2:
			data_out = (data[self.location[0]], data[self.location[1]])
		elif len(data) == 4:
			data_out = (data[self.location[0]], data[self.location[1]], data[self.location[2]], data[self.location[3]])
		else:
			raise ValueError(f"Data Dict Contain Unexpected # of Keys. Expected 2 or 4 keys, got {len(data)}")
		return data_out, interpipe

class Dict2TrainerBlock(DataFormattingBlock):
	"""
	Converts a dictionary to a trainer dictionary format.
	Accepts either 2 or 4 keys in the dictionary:
		- If 2 keys: 'neural' and 'behaviour', by default location (or no location).
		- If 4 keys: 'neural_train', 'neural_test', 'behaviour_train', 'behaviour_test', by default location (or no location).
	"""
	def __init__(self, location = None):
		super().__init__()
		self.location = location

	def transform(self, data, interpipe):
		"""
		Transform the data from a dictionary to a trainer dictionary format.
		Args:
			data (dict): Input data dictionary.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing either:
				- 'X' and 'Y' if 2 keys are present, (by default location).
				- 'X_train', 'X_val', 'Y_train', 'Y_val' if 4 keys are present
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if self.location is None:
			if len(data) == 2:
				self.location = ['neural', 'behaviour']
			elif len(data) == 4:
				self.location = ['neural_train', 'neural_test', 'behaviour_train', 'behaviour_test']

		if len(data) == 2:
			data_out = {'X': data[self.location[0]], 'Y': data[self.location[1]]}
		elif len(data) == 4:
			data_out = {'X_train': data[self.location[0]], 'X_val': data[self.location[1]], 'Y_train': data[self.location[2]], 'Y_val': data[self.location[3]]}
		else:
			raise ValueError(f"Data Dict Contain Unexpected # of Keys. Expected 2 or 4 keys, got {len(data)}")
		return data_out, interpipe

class Dataset2DictBlock(DataFormattingBlock):
	"""
	Converts a dictionary (from load_one_nwb) to neural and behaviour data in dictionary format.
	Add 'trial_idx' to interpipe.
	"""
	def __init__(self, neural_nwb_loc, behavior_nwb_loc, apply_trial_filtering = True, data_keys = ['neural', 'behaviour'], interpipe_keys = {'trial_start_times': 'trial_start_times', 'trial_end_times': 'trial_end_times', 'targets': 'targets'}, nwb_trial_start_times_loc = 'trials.cue_time', nwb_trial_end_times_loc = 'trials.stop_time', nwb_targets_loc = 'trials.target'):
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
		self.data_keys = data_keys
		self.interpipe_keys = interpipe_keys
		self.nwb_trial_start_times_loc = nwb_trial_start_times_loc
		self.nwb_trial_end_times_loc = nwb_trial_end_times_loc
		self.nwb_targets_loc = nwb_targets_loc
		super().__init__()

	def transform(self, data, interpipe):
		"""
		Transform the data from a dictionary to neural and behaviour data in dictionary format.
		Args:
			data (dict): Input dataset class.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing 'neural' and 'behaviour' data.
			interpipe (dict): Updated interpipe dictionary with entry of 'trial_idx' containing the trial indices.
		"""
		#TODO: Implement trial filtering (have an apply trial filters feature)
		neural, behaviour = resolve_path(data.dataset, self.neural_nwb_loc), resolve_path(data.dataset, self.behavior_nwb_loc)
		neural_ts, behaviour_ts = neural.timestamps[:] * 1000, behaviour.timestamps[:] * 1000
		neural, behaviour = neural.data[:], behaviour.data[:]
		trial_start_times = resolve_path(data.dataset, self.nwb_trial_start_times_loc)
		trial_end_times = resolve_path(data.dataset, self.nwb_trial_end_times_loc)
		targets = resolve_path(data.dataset, self.nwb_targets_loc)
		# Convert to milliseconds
		trial_start_times = trial_start_times[:] * 1000
		trial_end_times = trial_end_times[:] * 1000
		# Convert timestamps to milliseconds
		if self.apply_trial_filtering:
			UserWarning("Trial Filtering coming soon to a dataset near you")

		data_out = {self.data_keys[0]: neural, self.data_keys[1]: behaviour}
		interpipe[self.interpipe_keys['trial_start_times']] = trial_start_times
		interpipe[self.interpipe_keys['trial_end_times']] = trial_end_times
		interpipe[self.interpipe_keys['targets']] = targets[:]
		interpipe[f'{self.data_keys[0]}_ts'] = neural_ts
		interpipe[f'{self.data_keys[1]}_ts'] = behaviour_ts
		return data_out, interpipe

class IndexSelectorBlock(DataFormattingBlock):
	"""
	A block for selecting data from a dictionary based on indices.
	"""
	def __init__(self, location: List[str], indices: Union[int, list]):
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

class xdsDict2DictBlock(DataFormattingBlock):
	"""
	A block for converting an xds data dictionary to usable data dictionary.
	"""
	def __init__(self, channel_cutoff = 90, data_keys = ['neural', 'behaviour']):
		super().__init__()
		self.channel_cutoff = channel_cutoff # the xds code from limblab have variable (yes you saw it correct) number of channels,
											 # for the sake of consistency, we will use a fixed cutoff on channels.
		self.data_keys = data_keys

	def transform(self, data, interpipe):
		data_out = {}
		if(interpipe['is_train']):
			self.neural_key = 'day0_spike'
			self.behavior_key = 'day0_EMG'
		else:
			self.neural_key = 'dayk_spike'
			self.behavior_key = 'dayk_EMG'
		# Concatenate all trials in the list to create a single array
		data_out[self.data_keys[0]] = np.concatenate(data.dataset[self.neural_key], axis=0)
		data_out[self.data_keys[1]] = np.concatenate(data.dataset[self.behavior_key], axis=0)
		# Cutoff channels if necessary
		if data_out[self.data_keys[0]].shape[1] > self.channel_cutoff:
			data_out[self.data_keys[0]] = data_out[self.data_keys[0]][:, :self.channel_cutoff]
		if data_out[self.data_keys[1]].shape[1] > self.channel_cutoff:
			data_out[self.data_keys[1]] = data_out[self.data_keys[1]][:, :self.channel_cutoff]
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
	
class AddHistoryBlockRR(DataProcessingBlock):
	"""
	A block for adding history to the data at specified locations. For Ridge Regression model / linear models
	It uses `neuraldecoding.utils.add_hist` to add history.
	Keeps 2d shape
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
		data[self.location[0]], data[self.location[1]] = neuraldecoding.utils.add_hist(data[self.location[0]], data[self.location[1]], self.seq_length)
		return data, interpipe
	
class NormalizationBlock(DataProcessingBlock):
	def __init__(self, location, method, normalizer_params):
		super().__init__()
		self.location = location
		self.normalizer_method = method
		self.normalizer_params = normalizer_params
	
	def transform(self, data, interpipe):
		if self.normalizer_params['is_IO']:
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
			else:
				if 'save_path' not in self.normalizer_params:
					raise ValueError("NormalizationBlock requires 'save_path' in normalizer_params when is_save is True.")
				with open(self.normalizer_params['save_path'], 'rb') as f:
					normalizer = pickle.load(f)
				for loc in self.location:
					data[loc] = normalizer.transform(data[loc])
			return data, interpipe
		else:
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
		for loc in self.location_data:
			if loc not in data:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
		for loc in self.location_ts:
			if loc not in interpipe:
				raise ValueError(f"Location '{loc}' not found in interpipe dictionary.")
		
		# Extract binned features with full metadata (return_array=False)
		bin_features = self.feature_extractor.extract_binned_features(
			data=[data[loc] for loc in self.location_data], 
			timestamps_ms=[interpipe[loc] for loc in self.location_ts], 
			return_array=False
		)
		
		# Extract feature arrays and update data
		if len(self.location_data) > 1:
			# Multiple data streams - features is a list of arrays per bin
			features_per_stream = [[] for _ in self.location_data]
			for bin_feat in bin_features:
				for i, stream_features in enumerate(bin_feat['features']):
					features_per_stream[i].append(stream_features)
			
			# Convert to numpy arrays and update data
			for i, loc in enumerate(self.location_data):
				data[loc] = np.array(features_per_stream[i])
		else:
			# Single data stream - features is a single array per bin
			feature_arrays = [bin_feat['features'] for bin_feat in bin_features]
			data[self.location_data[0]] = np.array(feature_arrays)
		
		# Extract bin center timestamps and update interpipe
		bin_timestamps = np.array([bin_feat['bin_center_ms'] for bin_feat in bin_features])
		
		# Update timestamps for all location_ts entries with the new binned timestamps
		for ts_loc in self.location_ts:
			interpipe[ts_loc] = bin_timestamps
		
		return data, interpipe
	

class RawToXPC(DataProcessingBlock):
	"""
	Downsample a raw signal by rectifying and summing over contiguous windows of n samples,
	where n = sampling_rate_hz / 1000. Also updates timestamps to the average of each window.

	- Input data is expected to be shaped as (time, channels) or (time,).
	- Timestamps in interpipe at `location_ts` must have length equal to the time dimension.
	- Output replaces the data at `location_data` (or `output_location` if provided) and
	  updates the timestamps at `location_ts` (or `output_ts_key`).
	"""
	def __init__(self, location_data: str, location_ts: str, sampling_rate_hz: float = 2000.0, output_location: str = None, output_ts_key: str = None):
		super().__init__()
		self.location_data = location_data
		self.location_ts = location_ts
		self.sampling_rate_hz = float(sampling_rate_hz)
		self.output_location = output_location if output_location is not None else location_data
		self.output_ts_key = output_ts_key if output_ts_key is not None else location_ts

	def transform(self, data, interpipe):
		# Validate inputs
		if self.location_data not in data:
			raise ValueError(f"RawToXPC requires '{self.location_data}' in data dictionary.")
		if self.location_ts not in interpipe:
			raise ValueError(f"RawToXPC requires '{self.location_ts}' in interpipe dictionary.")

		samples = data[self.location_data]
		times = interpipe[self.location_ts]

		# Convert to numpy arrays
		if isinstance(samples, torch.Tensor):
			samples_np = samples.detach().cpu().numpy()
		else:
			samples_np = np.asarray(samples)
		times_np = np.asarray(times).reshape(-1)

		if samples_np.ndim == 1:
			samples_np = samples_np.reshape(-1, 1)

		if samples_np.shape[0] != times_np.shape[0]:
			raise ValueError(
				f"Length mismatch between data time dimension ({samples_np.shape[0]}) and timestamps ({times_np.shape[0]})."
			)

		# Determine window size n = fs / 1000
		n_float = self.sampling_rate_hz / 1000.0
		n = int(round(n_float))
		if not np.isclose(n_float, n):
			warnings.warn(
				f"sampling_rate_hz/1000 is not an integer ({n_float:.6f}); rounding to nearest integer n={n}.",
				UserWarning,
			)
		if n <= 0:
			raise ValueError("Computed window size n must be positive.")

		T = samples_np.shape[0]
		M = T // n
		if M == 0:
			raise ValueError(
				f"Input has only {T} samples which is insufficient for a single window of size n={n}."
			)
		trim_T = M * n
		if trim_T != T:
			warnings.warn(
				f"Discarding last {T - trim_T} samples that do not fit into {n}-sample windows.",
				UserWarning,
			)

		# Trim to full windows and reshape
		s_trim = samples_np[:trim_T, :]
		t_trim = times_np[:trim_T]

		s_reshaped = s_trim.reshape(M, n, samples_np.shape[1])
		t_reshaped = t_trim.reshape(M, n)

		# Rectify and sum per window; average timestamps per window
		downsampled = np.abs(s_reshaped).sum(axis=1)  # (M, channels)
		t_downsampled = t_reshaped.mean(axis=1)       # (M,)

		# Write back
		data[self.output_location] = downsampled
		interpipe[self.output_ts_key] = t_downsampled

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
	
 
class MovementOnsetDetectionBlock(DataProcessingBlock):
	"""
	A block for detecting movement onset in the EMG data.
	"""
	def __init__(self, location_emg: str, location_times:str , detection_config: dict, neural_indices: list = None, output_key: str = 'onset_indices'):
		super().__init__()
		self.location_emg = location_emg
		self.location_times = location_times
		self.detection_config = detection_config
		self.neural_indices = neural_indices
		self.output_key = output_key
		self.movement_onset_detection = MovementOnsetDetector(detection_config)

	def transform(self, data, interpipe):
		"""
		Transform the data by detecting movement onset in the EMG data.
		"""
		# Grab timestamps and EMG
		emg = data[self.location_emg]
		times = interpipe[self.location_times]
		trial_start_times = interpipe['trial_start_times']
		trial_end_times = interpipe['trial_end_times']

		# Select specific neural channels if indices are provided
		if self.neural_indices is not None:
			emg = emg[:, self.neural_indices]

		# Detect movement onsets
		onsets = self.movement_onset_detection.detect_movement_onsets(emg, times, trial_start_times, trial_end_times)

		# Add onsets to data dictionary
		interpipe[self.output_key] = onsets

		# Plot all channels together with onset markers
		n_channels = emg.shape[1]
		fig, ax = plt.subplots(1, 1, figsize=(12, 6))
		
		# Plot all EMG channels
		for ch in range(n_channels):
			ax.plot(times, emg[:, ch], alpha=0.7, linewidth=0.8, label=f'Channel {ch}')
		
		# Add vertical lines for onsets
		for onset_time in onsets:
			if onset_time is not None and not np.isnan(onset_time):
				ax.axvline(x=onset_time, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Onset' if onset_time == onsets[0] else "")
		
		# Add trial boundaries for context
		for i, start_time in enumerate(trial_start_times):
			ax.axvline(x=start_time, color='green', linestyle=':', alpha=0.5, linewidth=1.5, 
					  label='Trial Start' if i == 0 else "")
		for i, end_time in enumerate(trial_end_times):
			ax.axvline(x=end_time, color='orange', linestyle=':', alpha=0.5, linewidth=1.5, 
					  label='Trial End' if i == 0 else "")
		
		ax.set_xlabel('Time (ms)')
		ax.set_ylabel('EMG Amplitude')
		ax.set_title('EMG Channels with Movement Onsets')
		ax.grid(True, alpha=0.3)
		ax.legend()
		plt.tight_layout()
		plt.show(block=True)

		return data, interpipe

class TemplateBehaviorReplacementBlock(DataProcessingBlock):
	"""
	A block for replacing the behavior data with a template behavior.
	"""
	def __init__(self, location_behavior: str, location_out: str, location_onsets: str, template_config: dict, kinematic_indices: list = None):
		super().__init__()
		self.location_behavior = location_behavior
		self.location_out = location_out
		self.location_onsets = location_onsets
		self.template_config = template_config
		self.kinematic_indices = kinematic_indices
	
	def transform(self, data, interpipe):
		"""
		Transform the data by replacing the behavior data with a template behavior.
		"""
		# Get behavior data and timestamps
		kinematics = data[self.location_behavior]
		behavior_ts = interpipe.get(f'{self.location_behavior}_ts', interpipe.get('behavior_ts'))
		
		if behavior_ts is None:
			raise ValueError(f"Could not find timestamps for behavior data. Expected '{self.location_behavior}_ts' or 'behavior_ts' in data.")
		
		# Get onsets from interpipe
		movement_onsets = interpipe[self.location_onsets]
		
		# Get trial timing information from interpipe
		trial_start_times = interpipe['trial_start_times']
		trial_end_times = interpipe['trial_end_times']
		
		# Get targets and other parameters from template config
		targets = interpipe['targets']
		template_type = self.template_config.get('template_type', 'sigmoid')
		
		# Convert behavior data to numpy array if not already
		if not isinstance(kinematics, np.ndarray):
			kinematics = np.array(kinematics, dtype=np.float32)
		
		# Index the second axis of kinematics if kinematic_indices is provided
		if self.kinematic_indices is not None:
			kinematics = kinematics[:, self.kinematic_indices]
			targets = targets[:, self.kinematic_indices]
		
		# Apply template behavior
		templated_kinematics = self._apply_template_kinematics(
			kinematics=kinematics,
			behavior_ts=behavior_ts,
			trial_start_times=trial_start_times,
			trial_end_times=trial_end_times,
			movement_onsets=movement_onsets,
			targets=targets,
			template_type=template_type,
			template_params=self.template_config.get('template_params', {})
		)
		
		# Plot original vs templated kinematics
		self._plot_kinematics_comparison(
			original_kinematics=kinematics,
			templated_kinematics=templated_kinematics,
			behavior_ts=behavior_ts,
			trial_start_times=trial_start_times,
			trial_end_times=trial_end_times,
			movement_onsets=movement_onsets,
			kinematic_indices=self.kinematic_indices
		)
		
		# Update the behavior data in the data dictionary
		if self.kinematic_indices is not None:
			if self.location_out in data:
				data[self.location_out][:, self.kinematic_indices] = templated_kinematics
			else:
				# Copy input kinematics to output location and then modify the desired indices
				data[self.location_out] = kinematics.copy()
				data[self.location_out][:, self.kinematic_indices] = templated_kinematics
		else:
			data[self.location_out] = templated_kinematics
		
		return data, interpipe
	
	def _apply_template_kinematics(
		self,
		kinematics: np.ndarray,
		behavior_ts: np.ndarray,
		trial_start_times: np.ndarray,
		trial_end_times: np.ndarray,
		movement_onsets: np.ndarray,
		targets: np.ndarray,
		template_type: str = 'sigmoid',
		template_params: dict = None
	) -> np.ndarray:
		"""
		Apply template kinematics based on movement onsets and targets.

		Args:
			kinematics: np.ndarray - Array of shape (T, N) where N is the number of position variables
			behavior_ts: np.ndarray - Array of behavior timestamps (must be sorted)
			trial_start_times: np.ndarray - Array of trial start times
			trial_end_times: np.ndarray - Array of trial end times
			movement_onsets: np.ndarray - Array of shape (n_trials,) containing onset times for each trial
			targets: np.ndarray - Array of shape (n_trials, D) containing target positions for each trial
			template_type: str - Type of template to apply ('sigmoid', 'linear', 'step', etc.)
			template_params: dict - Additional parameters for the template function

		Returns:
			np.ndarray - Templated kinematics array of the same shape
		"""
		if template_params is None:
			template_params = {}
			
		templated = kinematics.copy()

		# All columns are position data
		N = kinematics.shape[1]
		
		# Find trial boundaries using searchsorted
		trial_start_indices = np.searchsorted(behavior_ts, trial_start_times, side='left')
		trial_end_indices = np.searchsorted(behavior_ts, trial_end_times, side='right')

		for trial_idx, (trial_start_idx, trial_end_idx) in enumerate(zip(trial_start_indices, trial_end_indices)):
			# Ensure indices are within bounds
			trial_start_idx = max(0, trial_start_idx)
			trial_end_idx = min(len(behavior_ts), trial_end_idx)
			
			if trial_start_idx >= trial_end_idx:
				continue  # Skip empty trials

			for i in range(N):  # Apply to all position dimensions
				# Check if we have an onset for this trial
				if trial_idx >= len(movement_onsets) or movement_onsets[trial_idx] is None or np.isnan(movement_onsets[trial_idx]):
					# If we don't know the onset, keep original kinematics in trial
					continue
				
				# Find onset index within this trial using the onset time
				onset_time = movement_onsets[trial_idx]
				onset_idx = np.searchsorted(behavior_ts[trial_start_idx:trial_end_idx], onset_time, side='left')
				onset_idx = trial_start_idx + onset_idx
				onset_idx = np.clip(onset_idx, trial_start_idx, trial_end_idx - 1)
				
				# Get target for this trial and dimension
				if trial_idx >= len(targets) or i >= targets.shape[1]:
					continue
				target = targets[trial_idx, i]
				
				# Set constant value before onset
				templated[trial_start_idx:onset_idx+1, i] = (
					templated[trial_start_idx - 1, i]
					if trial_start_idx > 0
					else templated[trial_start_idx, i]
				)
				
				# Get initial value at onset
				initial_value = templated[onset_idx, i]
				
				if trial_end_idx <= onset_idx:
					continue  # No samples after onset
				
				# Apply template from onset to end of trial
				num_samples = trial_end_idx - onset_idx
				duration_s = (behavior_ts[trial_end_idx - 1] - behavior_ts[onset_idx]) / 1000  # Convert ms to seconds
				
				template_values = self._generate_template(
					template_type=template_type,
					num_samples=num_samples,
					initial_value=initial_value,
					final_value=target,
					duration_s=duration_s,
					**template_params
				)
				
				templated[onset_idx:trial_end_idx, i] = template_values

		return templated
	
	def _generate_template(
		self,
		template_type: str,
		num_samples: int,
		initial_value: float,
		final_value: float,
		duration_s: float,
		**kwargs
	) -> np.ndarray:
		"""
		Generate template values based on the specified template type.
		
		Args:
			template_type: Type of template ('sigmoid', 'linear', 'step', 'exponential')
			num_samples: Number of samples to generate
			initial_value: Starting value
			final_value: Target value
			duration_s: Duration in seconds
			**kwargs: Additional parameters specific to each template type
		
		Returns:
			np.ndarray: Array of template values
		"""
		t = np.linspace(0, 1, num_samples)
		
		if template_type == 'sigmoid':
			return self._sigmoid_template(t, initial_value, final_value, duration_s, **kwargs)
		elif template_type == 'linear':
			return self._linear_template(t, initial_value, final_value)
		elif template_type == 'step':
			return self._step_template(t, initial_value, final_value, **kwargs)
		elif template_type == 'exponential':
			return self._exponential_template(t, initial_value, final_value, **kwargs)
		else:
			raise ValueError(f"Unknown template type: {template_type}")
	
	def _sigmoid_template(
		self,
		t: np.ndarray,
		initial_value: float,
		final_value: float,
		duration_s: float,
		steepness: float = 10,
		start_point: float = 0,
		start_threshold_percentage: float = 0.005,
		**kwargs
	) -> np.ndarray:
		"""Sigmoid template function."""
		# Normalized steepness
		s_norm = steepness * duration_s
		# Amplitude
		amplitude = final_value - initial_value
		# Calculate the normalized time midpoint t0_norm using s_norm
		log_arg = start_threshold_percentage / (1 - start_threshold_percentage)
		if log_arg <= 0:
			raise ValueError("Logarithm argument non-positive.")
		logit_val = np.log(log_arg)
		t0_norm = start_point - (1 / s_norm) * logit_val
		# Calculate the sigmoid value(s) using normalized time and s_norm
		exponent = -s_norm * (t - t0_norm)
		sigmoid_val = 1 / (1 + np.exp(exponent))
		return initial_value + amplitude * sigmoid_val
	
	def _linear_template(self, t: np.ndarray, initial_value: float, final_value: float) -> np.ndarray:
		"""Linear template function."""
		return initial_value + (final_value - initial_value) * t
	
	def _step_template(self, t: np.ndarray, initial_value: float, final_value: float, step_time: float = 0.5, **kwargs) -> np.ndarray:
		"""Step template function."""
		values = np.full_like(t, initial_value)
		values[t >= step_time] = final_value
		return values
	
	def _exponential_template(self, t: np.ndarray, initial_value: float, final_value: float, time_constant: float = 0.3, **kwargs) -> np.ndarray:
		"""Exponential template function."""
		amplitude = final_value - initial_value
		return initial_value + amplitude * (1 - np.exp(-t / time_constant))

	def _plot_kinematics_comparison(
		self,
		original_kinematics: np.ndarray,
		templated_kinematics: np.ndarray,
		behavior_ts: np.ndarray,
		trial_start_times: np.ndarray,
		trial_end_times: np.ndarray,
		movement_onsets: np.ndarray,
		kinematic_indices: list = None
	):
		"""
		Plot comparison between original and templated kinematics.
		
		Args:
			original_kinematics: Original kinematic data
			templated_kinematics: Templated kinematic data  
			behavior_ts: Behavior timestamps
			trial_start_times: Trial start times
			trial_end_times: Trial end times
			movement_onsets: Movement onset times
			kinematic_indices: Indices of kinematic dimensions to plot
		"""
		# Determine which dimensions to plot
		# Note: original_kinematics and templated_kinematics are already sliced if kinematic_indices was provided
		n_dims = original_kinematics.shape[1]
		
		if kinematic_indices is not None:
			# Data is already sliced, so plot indices 0, 1, 2, ... but label with original indices
			plot_indices = list(range(n_dims))
			plot_labels = [f'Dimension {kinematic_indices[i]}' for i in range(n_dims)]
		else:
			# Data contains all dimensions
			plot_indices = list(range(n_dims))
			plot_labels = [f'Dimension {i}' for i in plot_indices]
		
		# Create subplots - one for each kinematic dimension
		fig, axes = plt.subplots(n_dims, 1, figsize=(14, 4*n_dims), sharex=True)
		if n_dims == 1:
			axes = [axes]  # Make it iterable for single subplot
		
		fig.suptitle('Original vs Templated Kinematics Comparison', fontsize=16, fontweight='bold')
		
		for i, (dim_idx, ax, label) in enumerate(zip(plot_indices, axes, plot_labels)):
			# Plot original kinematics
			ax.plot(behavior_ts, original_kinematics[:, dim_idx], 
				   color='blue', alpha=0.7, linewidth=1.5, label='Original')
			
			# Plot templated kinematics
			ax.plot(behavior_ts, templated_kinematics[:, dim_idx], 
				   color='red', alpha=0.8, linewidth=2, label='Templated')
			
			# Add trial boundaries
			for j, start_time in enumerate(trial_start_times):
				ax.axvline(x=start_time, color='green', linestyle=':', alpha=0.6, linewidth=1,
						  label='Trial Start' if i == 0 and j == 0 else "")
			
			for j, end_time in enumerate(trial_end_times):
				ax.axvline(x=end_time, color='orange', linestyle=':', alpha=0.6, linewidth=1,
						  label='Trial End' if i == 0 and j == 0 else "")
			
			# Add movement onsets
			for j, onset_time in enumerate(movement_onsets):
				if onset_time is not None and not np.isnan(onset_time):
					ax.axvline(x=onset_time, color='purple', linestyle='--', alpha=0.8, linewidth=2,
							  label='Movement Onset' if i == 0 and j == 0 else "")
			
			# Formatting
			ax.set_ylabel(f'{label}\nPosition')
			ax.grid(True, alpha=0.3)
			ax.legend(loc='upper right')
			
			# Set title for each subplot
			ax.set_title(f'{label} - Original vs Templated', fontweight='bold')
		
		# Set common x-label
		axes[-1].set_xlabel('Time (ms)')
		
		plt.tight_layout()
		plt.show(block=True)


class ReFITTransformationBlock(DataProcessingBlock):

	def __init__(self, location_data: str,location_data_ts:str, location_targets: str, location_out: str, location_trial_start_times: str, location_trial_end_times: str, target_radius: float = 0.075):
		self.location_data = location_data
		self.location_data_ts = location_data_ts
		self.location_targets = location_targets
		self.location_out = location_out
		self.location_trial_start_times = location_trial_start_times
		self.location_trial_end_times = location_trial_end_times
		self.target_radius = target_radius

	def transform(self, data, interpipe):
		targets = interpipe[self.location_targets]
		trial_start_times = interpipe[self.location_trial_start_times]
		trial_end_times = interpipe[self.location_trial_end_times]
		
		kinematics = data[self.location_data]
		kinematics_ts = interpipe[self.location_data_ts]
		# Assume kinematics has 2*n columns, first n are position, last n are velocity
		Ndofs = kinematics.shape[1] // 2
		pos = kinematics[:, :Ndofs]
		vel = kinematics[:, Ndofs:]
		# Apply ReFIT transformation
		transformed_vel = self._apply_refit_transformation(vel, pos, targets, kinematics_ts, trial_start_times, trial_end_times)
		# Combine transformed pos and vel
		transformed_kinematics = np.concatenate([pos, transformed_vel], axis=1)
		data[self.location_out] = transformed_kinematics
		return data, interpipe

	def _apply_refit_transformation(self, vel: np.ndarray, pos: np.ndarray, targets: np.ndarray, kinematics_ts: np.ndarray, trial_start_times: np.ndarray, trial_end_times: np.ndarray):
		"""
		Apply ReFIT transformation to velocity data.
		ReFIT rotates velocity vectors to point towards the target while preserving magnitude.
		Sets velocity to zero when position is within target_radius of the target.
		
		Args:
			vel: Velocity data with shape (time_points, n_dofs)
			pos: Position data with shape (time_points, n_dofs)
			targets: Target positions with shape (n_trials, n_dofs)
			kinematics_ts: Timestamps for kinematic data
			trial_start_times: Start times for each trial
			trial_end_times: End times for each trial
		
		Returns:
			Transformed velocity data with same shape as input
		"""
		# Get dimensions
		n_timepoints, n_dofs = vel.shape
		n_trials = len(targets)
		
		# Create time-aligned target vector
		target_vector = np.zeros((n_timepoints, n_dofs))
		
		# For each trial, find the time indices that belong to it and assign the target
		for trial_idx in range(n_trials):
			# Find timestamps that fall within this trial
			trial_mask = (kinematics_ts >= trial_start_times[trial_idx]) & (kinematics_ts <= trial_end_times[trial_idx])
			
			# Assign the target for this trial to all timepoints within the trial
			target_vector[trial_mask, :] = targets[trial_idx, :]
		
		# Apply ReFIT transformation: rotate velocity to point towards target
		transformed_vel = vel.copy()
		
		for t in range(n_timepoints):
			current_vel = vel[t, :]
			current_pos = pos[t, :]
			current_target = target_vector[t, :]
			
			# Check if position is within target radius
			distance_to_target = np.linalg.norm(current_pos - current_target)
			if distance_to_target <= self.target_radius:
				# Set velocity to zero when within target radius
				transformed_vel[t, :] = 0.0
				continue
			
			# Skip if velocity is zero
			if np.allclose(current_vel, 0):
				continue
			
			# Calculate velocity magnitude (preserve this)
			vel_magnitude = np.linalg.norm(current_vel)
			
			# Calculate target direction (unit vector towards target)
			target_direction = (current_target - current_pos)
			target_magnitude = np.linalg.norm(target_direction)
			if target_magnitude > 1e-8:
				target_direction = target_direction / target_magnitude
				
				# Rotate velocity: preserve magnitude but point towards target
				transformed_vel[t, :] = vel_magnitude * target_direction
		
		return transformed_vel
		
class TrimAndSmoothBlock(DataProcessingBlock):
    """
    A block for trimming neural and kinematic (finger) data around movement onset,
    determined by deviations from baseline, and then smoothing the neural data.
    """

    def __init__(self, neural_key="neural_train", finger_key="finger_train",
                 std_multiplier=2, sigma=2.5, pre_ms=120, post_ms=420, bin_ms=20):
        """
        Initializes the TrimBlock.

        Args:
            neural_key (str): Key in the data dict for neural data.
            finger_key (str): Key in the data dict for kinematic data.
            std_multiplier (float): Threshold multiplier for std above baseline.
            sigma (float): Gaussian smoothing sigma (in samples -> calculated by 50ms/bin_ms, where 50 ms is from Gallego 2020 paper)
            pre_ms (int): How many ms before movement onset to include.
            post_ms (int): How many ms after movement onset to include.
            bin_ms (int): Bin width in ms.
        """
        super().__init__()
        self.neural_key = neural_key
        self.finger_key = finger_key
        self.std_multiplier = std_multiplier
        self.sigma = sigma
        self.pre_bins = pre_ms // bin_ms
        self.post_bins = post_ms // bin_ms

    def transform(self, data, interpipe):
        """
        Trim trials around movement onset and smooth neural data.

        Args:
            data (dict): Dictionary with keys [self.neural_key, self.finger_key].
                         - neural: list/array of shape (n_trials, T, C)
                         - finger: list/array of shape (n_trials, T)
            interpipe (dict): Shared pipeline dictionary.

        Returns:
            data (dict): With trimmed + smoothed neural/finger data.
            interpipe (dict): Unchanged.
        """
        neural_trials = data[self.neural_key]
        finger_trials = data[self.finger_key]

        trimmed_neural = []
        trimmed_finger = []

        for trial_idx, (trial_neural, trial_finger) in enumerate(zip(neural_trials, finger_trials)):
            # baseline = first 10% of trial
            baseline_period = trial_finger[: int(0.1 * len(trial_finger))]
            baseline_mean = np.mean(baseline_period)
            baseline_std = np.std(baseline_period)

            threshold = baseline_mean + self.std_multiplier * baseline_std
            min_threshold = baseline_mean - self.std_multiplier * baseline_std

            # movement onset idx
            movement_onset_idx = next(
                (i for i, val in enumerate(trial_finger)
                 if (val > threshold) or (val < min_threshold)), 0
            )

            start = movement_onset_idx - self.pre_bins
            end = movement_onset_idx + self.post_bins

            if start < 0 or end >= trial_neural.shape[0]:
                continue  # too short, skip trial

            # trim and smooth
            neural_trimmed = trial_neural[start:end, :]
            neural_smoothed = gaussian_filter1d(neural_trimmed, sigma=self.sigma, axis=0)

            finger_trimmed = trial_finger[start:end]

            trimmed_neural.append(neural_smoothed)
            trimmed_finger.append(finger_trimmed)


        # Replace data dict
        data[self.neural_key] = np.array(trimmed_neural, dtype=object)
        data[self.finger_key] = np.array(trimmed_finger, dtype=object)

        return data, interpipe