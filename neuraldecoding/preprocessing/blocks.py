from ..dataaugmentation import SequenceScaler
from ..feature_extraction import FeatureExtractor
from ..utils.utils_general import resolve_path
from ..utils.data_tools import load_one_nwb
import sklearn.preprocessing
from .. import utils
from .. import stabilization

import torch

from abc import ABC, abstractmethod

import time
import pickle
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
				- (neural, behavior) , contains either training or testing data depending on 'is_train' in interpipe.
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

class DataSplitBlock(DataFormattingBlock):
	"""
	A block for splitting data into training and testing sets based on trial indices.
	Assumes the data dictionary contains 'neural' and 'behavior' keys, and the interpipe dictionary contains 'trial_idx'.
	It uses `neuraldecoding.utils.data_split_trial` to perform the split.
	"""
	def __init__(self, split_ratio: 0.8, split_seed: 42):
		"""
		Initializes the DataSplitBlock.
		Args:
			split_ratio (float, tuple): The ratio of training data to total data. Default is 0.8. If tuple, 
				will be in the form of [train, validation, test] as a fraction of 1 (e.g. [0.7, 0.1, 0.2]).
			split_seed (int): Seed for random number generator to ensure reproducibility. Default is 42.
		"""
		super().__init__()
		self.split_ratio = split_ratio
		self.split_seed = split_seed

	def transform(self, data, interpipe):
		"""
		Transform the data by splitting it into training and testing sets based on trial indices.
		Assumes the data dictionary contains 'neural' and 'behavior' keys, and the interpipe dictionary contains 'trial_idx'.
		It uses `neuraldecoding.utils.data_split_trial` to perform the split.
		Args:
			data (dict): Input data dictionary containing 'neural' and 'behavior' keys.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing:
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

		data = utils.data_split_trial(data['neural'], 
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
				dataset = utils.datasets.BehaviorDataset(data, xkey, ykey)
			else:
				dataset = utils.datasets.BehaviorDatasetCustom(data, interpipe, xkey, ykey, otherdatakeys)
			datasets += (dataset,)
		
		return datasets, interpipe

class Dataset2DictBlock(DataFormattingBlock):
	"""
	Converts a dictionary (from load_one_nwb) to neural and behavior data in dictionary format.
	Add 'trial_idx' to interpipe.
	"""
	def __init__(self, neural_nwb_loc, behavior_nwb_loc, apply_trial_filtering = True, nwb_trial_start_times_loc = 'trials.cue_time', nwb_trial_end_times_loc = 'trials.stop_time', nwb_targets_loc = 'trials.target'):
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
		self.nwb_trial_start_times_loc = nwb_trial_start_times_loc
		self.nwb_trial_end_times_loc = nwb_trial_end_times_loc
		self.nwb_targets_loc = nwb_targets_loc
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
		trial_start_times = trial_start_times[:] 
		trial_end_times = trial_end_times[:]
		# Convert timestamps to milliseconds
		neural_ts, behavior_ts = neural_nwb.timestamps[:] * 1000, behavior_nwb.timestamps[:] * 1000
		if self.apply_trial_filtering:
			UserWarning("Trial Filtering coming soon to a dataset near you")

		trial_idx = np.searchsorted(neural_ts, trial_start_times, side='left')

		data_out = {'neural': neural, 'neural_ts': neural_ts, 'behavior': behavior, 'behavior_ts': behavior_ts}
		interpipe['trial_start_times'] = trial_start_times
		interpipe['trial_end_times'] = trial_end_times
		interpipe['trial_idx'] = trial_idx
		interpipe['targets'] = targets[:]
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
		stabilization_method = getattr(stabilization.latent_space_alignment, stabilization_config["type"])
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
			data[loc] = utils.add_history_numpy(data[loc], self.seq_length)

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
			utils.add_trial_history(data['neural_train'], data['behavior_train'], trial_filt_train, self.leadup)
		data['trial_lengths_train'] = trial_lengths_train
		interpipe['leadup'] = self.leadup

		return data, interpipe


class NormalizationBlock(DataProcessingBlock):
	def __init__(self, fit_location, apply_locations, normalizer_method, 
			  normalizer_params, sklearn_type=None, save_path=None, retain_denorm=False):
		super().__init__()
		self.fit_location = fit_location
		if isinstance(apply_locations, str):
			apply_locations = [apply_locations]
		self.apply_location = apply_locations
		self.normalizer_method = normalizer_method
		self.normalizer_params = normalizer_params
		self.sklearn_type = sklearn_type
		self.save_path = save_path
		self.scaler = None
		self.retain_denorm = retain_denorm
		self.denorm_data = {}
	def transform(self, data, interpipe):
		if self.normalizer_method == 'sklearn':
			normalizer = getattr(sklearn.preprocessing, self.sklearn_type)(**self.normalizer_params)
			data[self.fit_location] = normalizer.fit_transform(data[self.fit_location])
			for loc in self.apply_location:
				if self.retain_denorm:
					self.denorm_data[loc] = data[loc].copy()
				data[loc] = normalizer.transform(data[loc])
		elif self.normalizer_method == 'sequence_scaler':	
			normalizer = SequenceScaler()
			data[self.fit_location] = normalizer.fit_transform(data[self.fit_location], **self.normalizer_params)
			for loc in self.apply_location:
				if self.retain_denorm:
					self.denorm_data[loc] = data[loc].copy()
				data[loc] = normalizer.transform(data[loc])
		else:
			raise ValueError(f"Unsupported normalization method: {self.normalizer_method}")
		if self.save_path is not None:
			with open(self.save_path, 'wb') as f:
				pickle.dump(normalizer, f)
		self.scaler = normalizer
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
		self.target_filter = feature_extractor_config.get('target_filter', None)
		if self.target_filter is not None:
			self.target_filter = np.array(self.target_filter, dtype=np.int32)

	def transform(self, data, interpipe):
		if self.target_filter is None:
			self.target_filter = np.arange(data['targets'].shape[1])
		for loc in self.location_data + self.location_ts:
			if loc not in data:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
		features_list = self.feature_extractor.extract_binned_features(data=[data[loc] for loc in self.location_data], timestamps_ms=[data[loc] for loc in self.location_ts], return_array=True)
		for loc in self.location_ts:
			del data[loc]
		for i, loc in enumerate(self.location_data):
			data[loc] = features_list[i]
		if 'trial_idx' in interpipe:
			interpipe['trial_idx'] = np.astype((interpipe['trial_idx'] / self.feature_extractor.bin_size_ms), np.int32)
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

	def __init__(self, nicknames, param_dict, retain_unmodified=False):
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
		self.nicknames = nicknames
		self.retain_unmodified = retain_unmodified
		self.unmodified_data = None
	
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
		if isinstance(self.nicknames, str):
			self.nicknames = [self.nicknames]
		
		if self.retain_unmodified:
			self.unmodified_data = data['behavior_train'].copy()

		data['behavior_train'] = utils.label_mods.apply_modifications(self.nicknames, data['behavior_train'], interpipe, self.param_dict)
		
		return data, interpipe
	
