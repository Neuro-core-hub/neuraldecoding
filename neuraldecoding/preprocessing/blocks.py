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
import numpy as np
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt


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
	
class IndexSelectorBlock(DataFormattingBlock):
	"""
	A block for selecting specific indices from the data dictionary.
	"""
	def __init__(self, locations, indices):
		super().__init__()
		if isinstance(locations, str):
			locations = [locations]
		self.locations = locations
		self.indices = indices

	def transform(self, data, interpipe):
		"""
		Transform the data by selecting features from the specified locations.
		Args:
			data (dict): Input data dictionary containing the data to be transformed.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): The data dictionary with selected indices at the specified locations.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		for loc in self.locations:
			if loc in data:
				data[loc] = data[loc][:, self.indices]
			else:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
		return data, interpipe

class ClassificationDict2TupleBlock(DataFormattingBlock):
	"""
	Converts a dictionary to a tuple format for classification tasks.
	Temporary block designed specifically for LDA and data from RPNI C-P2 experiments.

	Assumes the dictionary contains keys 'X_train', 'y_train', 'X_test', and 'y_test'.
	Returns a tuple of (neural, behavior) for training or testing based on the 'is_train' key in the interpipe dictionary.
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
			neural, behavior = (data['X_train'], data['y_train'].flatten())
		else:
			neural, behavior = (data['X_test'], data['y_test'].flatten())
		data_out = (neural, behavior)
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
	Assumes the data dictionary contains 'neural' and 'behavior' keys, and the interpipe dictionary contains 'trial_idx'.
	It uses `neuraldecoding.utils.data_split_trial` to perform the split.
	"""
	def __init__(self, split_ratio: 0.8, split_seed: 42, location = ['neural', 'behaviour'], interpipe_location = ['trial_idx'], data_keys = ['neural_train', 'neural_test', 'behaviour_train', 'behaviour_test']):
		"""
		Initializes the DataSplitBlock.
		Args:
			split_ratio (float, tuple): The ratio of training data to total data. Default is 0.8. If tuple, 
				will be in the form of [train, validation, test] as a fraction of 1 (e.g. [0.7, 0.1, 0.2]).
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

	def transform(self, data, interpipe):
		"""
		Transform the data by splitting it into training and testing sets based on trial indices.
		Assumes the data dictionary contains 'neural' and 'behavior' keys, and the interpipe dictionary contains 'trial_idx'.
		It uses `neuraldecoding.utils.data_split_trial` to perform the split.
		Args:
			data (dict): Input data dictionary containing 'neural' and 'behavior' keys.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing (by default data keys):
				- 'neural_train': Training neural data
				- 'neural_test': Testing neural data
				- 'behavior_train': Training behavior data
				- 'behavior_test': Testing behavior data
				if split_ratio is a tuple, also includes:
				- 'neural_val': Validation neural data
				- 'behavior_val': Validation behavior data
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if 'trial_idx' not in interpipe:
			raise ValueError("DataSplitBlock requires 'trial_idx' in interpipe from other wrappers (Dict2DataDictBlock).")

		data = neuraldecoding.utils.data_split_trial(data['neural'], 
										data['behavior'], 
										interpipe, 
										split_ratio=self.split_ratio, 
										seed=self.split_seed)

		return data, interpipe

class Dict2BehaviorDatasetBlock(DataFormattingBlock):
	"""
	Converts a dictionary to a BehaviorDataset or BehaviorDatasetCustom class.
	"""
	def __init__(self, otherdatakeys_list=None):
		"""
		Initializes the Dict2BehaviorDatasetBlock. Creates BehaviorDataset objects for train, validation (if included) and test. 
		BehaviorDatasetCustom allows for additional data keys to be included in the dataset. otherdatakeys_list can be list of lists. Ensure 
		that otherdatakeys_list and otherinterpipekeys_list have settings in the order of [train, valid, test] or [train, test] if no validation set is used.
		e.g. ,
			otherdatakeys_list = [['trial_lengths_train'], None]
		Args:
			otherdatakeys_list (str, list, optional): Additional locations in the data dictionary to include. Must be length 2 if 
				only train and test datasets are used, or length 3 if train, valid, and test datasets are used, or None.
			otherinterpipekeys_list (str, list, optional): Additional keys in the interpipe dictionary to include. Must be length 2 if
				only train and test datasets are used, or length 3 if train, valid, and test datasets are used, or None.
		"""
		super().__init__()

		self.otherdatakeys_list = otherdatakeys_list
	
	def transform(self, data, interpipe):
		"""
		Transform the data from a dictionary to BehaviorDataset or BehaviorDatasetCustom class.
		Args:
			data (dict): Input data dictionary containing the data and relevant information.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (tuple): A tuple of BehaviorDataset or BehaviorDatasetCustom instances in order of [train, valid, test] or [train, test].
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if 'neural_val' in data:
			self.xkey = ['neural_train', 'neural_val', 'neural_test']
			self.ykey = ['behavior_train', 'behavior_val', 'behavior_test']
		else:
			self.xkey = ['neural_train', 'neural_test']
			self.ykey = ['behavior_train', 'behavior_test']

		if self.otherdatakeys_list is None:
			self.otherdatakeys_list = [None] * len(self.xkey)
		
		if len(self.otherdatakeys_list) != len(self.xkey):
			raise ValueError("otherdatakeys_list size mismatch.")

		datasets = ()
		for xkey, ykey, otherdatakeys in zip(self.xkey, self.ykey, self.otherdatakeys_list):
			if otherdatakeys is None:
				dataset = neuraldecoding.utils.datasets.BehaviorDataset(data, xkey, ykey)
			else:
				dataset = neuraldecoding.utils.datasets.BehaviorDatasetCustom(data, interpipe, xkey, ykey, otherdatakeys)
			datasets += (dataset,)
		
		return datasets, interpipe

class Dataset2DictBlock(DataFormattingBlock):
	"""
	Converts a dictionary (from load_one_nwb) to neural and behavior data in dictionary format.
	Add 'trial_idx' to interpipe.
	"""
	def __init__(self, neural_nwb_loc, behavior_nwb_loc, apply_trial_filtering = True, continuous_time_scaler=None, trial_time_scaler=None,
			  nwb_trial_start_times_loc = 'trials.cue_time', nwb_trial_end_times_loc = 'trials.stop_time', nwb_targets_loc = 'trials.target'):
		"""
		Initializes the Dataset2DictBlock.
		Initializes the Dataset2DictBlock.
		Args:
			neural_nwb_loc (str): Location path for neural data in the NWB file.
			behavior_nwb_loc (str): Location path for behavior data in the NWB file.
			apply_trial_filtering (bool): Whether to apply trial filtering. Default is True.
			neural_nwb_loc (str): Location path for neural data in the NWB file.
			behavior_nwb_loc (str): Location path for behavior data in the NWB file.
			apply_trial_filtering (bool): Whether to apply trial filtering. Default is True.
		"""
		self.neural_nwb_loc = neural_nwb_loc
		self.behavior_nwb_loc = behavior_nwb_loc
		self.apply_trial_filtering = apply_trial_filtering
		self.nwb_trial_start_times_loc = nwb_trial_start_times_loc
		self.nwb_trial_end_times_loc = nwb_trial_end_times_loc
		self.nwb_targets_loc = nwb_targets_loc
		if continuous_time_scaler is None:
			continuous_time_scaler = 1
		self.continuous_time_scaler = continuous_time_scaler
		if trial_time_scaler is None:
			trial_time_scaler = 1
		self.trial_time_scaler = trial_time_scaler

		super().__init__()

	def transform(self, data, interpipe):
		"""
		Transform the data from a dictionary to neural and behavior data in dictionary format.
		Args:
			data (dict): Input dataset class.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing 'neural' and 'behavior' data.
			interpipe (dict): Updated interpipe dictionary with entry of 'trial_idx' containing the trial indices.
		"""
		#TODO: Implement trial filtering (have an apply trial filters feature)
		neural_nwb, behavior_nwb = resolve_path(data.dataset, self.neural_nwb_loc), resolve_path(data.dataset, self.behavior_nwb_loc)
		neural, behavior = neural_nwb.data[:], behavior_nwb.data[:]
		trial_start_times = resolve_path(data.dataset, self.nwb_trial_start_times_loc)
		trial_end_times = resolve_path(data.dataset, self.nwb_trial_end_times_loc)
		targets = resolve_path(data.dataset, self.nwb_targets_loc)
		# Convert to milliseconds
		trial_start_times = trial_start_times[:]*self.trial_time_scaler
		trial_end_times = trial_end_times[:]*self.trial_time_scaler
		# Convert timestamps to milliseconds
		neural_ts, behavior_ts = neural_nwb.timestamps[:] * self.continuous_time_scaler, behavior_nwb.timestamps[:] * self.continuous_time_scaler
		if self.apply_trial_filtering:
			UserWarning("Trial Filtering coming soon to a dataset near you")

		data_out = {'neural': neural, 'neural_ts': neural_ts, 'behavior': behavior, 'behavior_ts': behavior_ts}
		interpipe['trial_start_times'] = trial_start_times
		interpipe['trial_end_times'] = trial_end_times
		interpipe['targets'] = targets[:]
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

class TrialHistoryBlock(DataProcessingBlock):
	"""
	A block to add history but have each sequence be the length of an entire trial, plus padding and leadup.
	Uses 'add_trial_history' to add history.
	"""
	def __init__(self, leadup = 20):
		"""
		Initializes the TrialHistoryBlock.
		Args:
			leadup (int): The length of the history to be added before the first bin of each trial. Default is 20.
		"""
		self.set = set
		self.leadup = leadup

	def transform(self, data, interpipe):
		"""
		Transform the training data by adding entire trial history, plus padding and leadup.
		Args:
			data (dict): Input data dictionary containing the data to which history is added.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data (dict): The data dictionary with history added at the specified locations.
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		trial_filt_train = interpipe['trial_filt'][interpipe['train_mask']]
		data['neural_train'], data['behavior_train'], trial_lengths_train = \
			neuraldecoding.utils.add_trial_history(data['neural_train'], data['behavior_train'], trial_filt_train, self.leadup)
		data['trial_lengths_train'] = trial_lengths_train
		interpipe['leadup'] = self.leadup

		return data, interpipe


class NormalizationBlock(DataProcessingBlock):
	def __init__(self, fit_location, apply_locations, normalizer_method, 
			  normalizer_params, sklearn_type=None, save_path=None, save_denorm_data=False, save_normalizer=False):
		super().__init__()
		self.fit_location = fit_location
		if isinstance(apply_locations, str):
			apply_locations = [apply_locations]
		self.apply_location = apply_locations
		self.normalizer_method = normalizer_method
		self.normalizer_params = normalizer_params
		self.sklearn_type = sklearn_type
		self.save_path = save_path
		self.save_denorm_data = save_denorm_data
		self.save_normalizer = save_normalizer
	def transform(self, data, interpipe):
		if self.normalizer_method == 'sklearn':
			normalizer = getattr(sklearn.preprocessing, self.sklearn_type)(**self.normalizer_params)
			if self.save_denorm_data:
				denorm_key = 'denorm_' + self.fit_location
				interpipe[denorm_key] = data[self.fit_location]
				interpipe['save_keys'].append(denorm_key)
			data[self.fit_location] = normalizer.fit_transform(data[self.fit_location])
			for loc in self.apply_location:
				if self.save_denorm_data:
					denorm_key = 'denorm_' + loc
					interpipe[denorm_key] = data[loc]
					interpipe['save_keys'].append(denorm_key)
				data[loc] = normalizer.transform(data[loc])
		elif self.normalizer_method == 'sequence_scaler':	
			normalizer = SequenceScaler()
			data[self.fit_location] = normalizer.fit_transform(data[self.fit_location], **self.normalizer_params)
			if self.save_denorm_data:
				denorm_key = 'denorm_' + self.fit_location
				interpipe[denorm_key] = data[self.fit_location].copy()
				interpipe['save_keys'].append(denorm_key)
			for loc in self.apply_location:
				if self.save_denorm_data:
					denorm_key = 'denorm_' + loc
					interpipe[denorm_key] = data[loc]
				data[loc] = normalizer.transform(data[loc])
		else:
			raise ValueError(f"Unsupported normalization method: {self.normalizer_method}")
		if self.save_path is not None:
			with open(self.save_path, 'wb') as f:
				pickle.dump(normalizer, f)
		if self.save_normalizer:
			normalizer_key = 'normalizer_' + self.fit_location
			interpipe[normalizer_key] = normalizer
			interpipe['save_keys'].append(normalizer_key)
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
		self.location_data = location_data
		self.location_ts = location_ts
		self.feature_extractor = FeatureExtractor(feature_extractor_config)
		self.target_filter = feature_extractor_config.get('target_filter', None)
		if self.target_filter is not None:
			self.target_filter = np.array(self.target_filter, dtype=np.int32)

	def transform(self, data, interpipe):
		if self.target_filter is None:
			self.target_filter = np.arange(interpipe['targets'].shape[1])
		for loc in self.location_data + self.location_ts:
			if loc not in data:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
		trial_starts_ends = (interpipe['trial_start_times'], interpipe['trial_end_times'])
		features_list, interpipe['trial_idx'] = self.feature_extractor.extract_binned_features(
			data=[data[loc] for loc in self.location_data],
			timestamps_ms=[data[loc] for loc in self.location_ts],
			return_array=True,
			trial_starts_ends=trial_starts_ends
		)
		for loc in self.location_ts:
			del data[loc]
		for i, loc in enumerate(self.location_data):
			data[loc] = features_list[i]
			
		interpipe['trial_filt'] = np.zeros(len(data[self.location_data[0]]), dtype=np.int32)
		interpipe['targets_filt'] = np.zeros((len(data[self.location_data[0]]), self.target_filter.shape[0]), dtype=np.float32)
		for i, start in enumerate(interpipe['trial_idx']):
			end = interpipe['trial_idx'][i + 1] if i + 1 < len(interpipe['trial_idx']) else len(data[self.location_data[0]])
			interpipe['trial_filt'][start:end] = i
			interpipe['targets_filt'][start:end] = interpipe['targets'][i][self.target_filter]
		return data, interpipe
	
class LabelModificationBlock(DataProcessingBlock):
	"""
	A block to add label modifications to training data.
	"""

	def __init__(self, nicknames, param_dict, save_unmodified=False, save_name=None):
		"""
		Initializes the LabelModificationBlock. Below are modification options and the required parameters in param_dict.
		See the apply_modifications function in utils/label_mods.py function and hover over each individual modification 
		function to get a better sense of what they do.
		- 'shift_bins': 'shift'
		- 'shift_by_trial': 'shift_range', 'individuate_dofs'
		- 'warp_by_trial': 'warp_factor', 'hold_time'
		- 'random_warp': 'hold_time, 'individuate_dofs'
		- 'sigmoid_replacement': 'sigmoid_k', 'center'
		- 'bias_endpoints': 'bias_range', 'individuate_dofs'
		Args:
			modifications (str or list): modification or modifications to add to labels
			nicknames (str or list): nickname or nicknames for the modifications
			leadup (int): The length of the history to be added before the first bin of each trial. Default is 10.
		"""
		self.param_dict = param_dict
		if isinstance(nicknames, str):
			self.nicknames = [nicknames]
		else:
			self.nicknames = nicknames
		self.save_unmodified = save_unmodified
		self.save_name = save_name
	
	def transform(self, data, interpipe):
		"""
		Transform the data by modifying labels.
		Args:
			data (dict): Input data dictionary containing the data without history.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data (dict): The data dictionary with labels modified at the specified locations
			interpipe (dict): The interpipe dictionary remains unchanged
		"""		
		if self.save_unmodified:
			if self.save_name is None:
				curmod_num = 0
				while True:
					curmod_name = f"unmodified_{curmod_num}"
					if curmod_name not in interpipe['save_keys']:
						break
					curmod_num += 1
				self.save_name = curmod_name
			interpipe['save_keys'].append(self.save_name)
			interpipe[self.save_name] = data['behavior_train']

		data['behavior_train'] = neuraldecoding.utils.label_mods.apply_modifications(self.nicknames, data['behavior_train'], interpipe, self.param_dict)
		
		return data, interpipe

class SaveDataBlock(DataProcessingBlock):
	def __init__(self, locs, keys):
		super().__init__()
		if isinstance(locs, str):
			locs = [locs]
		if isinstance(keys, str):
			keys = [keys]
		if len(locs) != len(keys):
			raise ValueError("locs and keys must be the same length.")
		self.locs = locs
		self.keys = keys
	
	def transform(self, data, interpipe):
		for loc, key in zip(self.locs, self.keys):
			if loc not in data:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
			interpipe[key] = data[loc]
			interpipe['save_keys'].append(key)
			print(f"Saved data at location '{loc}' with key '{key}'.")
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
		behavior_ts = data.get(f'{self.location_behavior}_ts', data.get('behavior_ts'))
		
		if behavior_ts is None:
			raise ValueError(f"Could not find timestamps for behavior data. Expected '{self.location_behavior}_ts' or 'behavior_ts' in data.")
		
		# Get onsets from interpipe
		movement_onsets = interpipe[self.location_onsets]
		
		# Get trial timing information from interpipe
		trial_start_times = interpipe['trial_start_times']
		trial_end_times = interpipe['trial_end_times']
		
		# Get targets and other parameters from template config
		targets = data['targets']
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
