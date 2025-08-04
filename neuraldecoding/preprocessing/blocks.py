import neuraldecoding.utils
import neuraldecoding.utils.label_mods
import neuraldecoding.stabilization.latent_space_alignment
import neuraldecoding.dataaugmentation.DataAugmentation
from neuraldecoding.dataaugmentation import SequenceScaler
from neuraldecoding.feature_extraction import FeatureExtractor
from neuraldecoding.utils.utils_general import resolve_path
import sklearn.preprocessing
import neuraldecoding.utils.datasets

import torch

from abc import ABC, abstractmethod

import time
import pickle

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
		(neural, finger), trial_idx, trial_ts = neuraldecoding.utils.neural_finger_from_dict(data, self.neural_type)
		interpipe['trial_idx'] = trial_idx
		interpipe['trial_ts'] = trial_ts

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
	A block for splitting data into training and testing sets based on trial indices.
	Assumes the data dictionary contains 'neural' and 'finger' keys, and the interpipe dictionary contains 'trial_idx'.
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
		Assumes the data dictionary contains 'neural' and 'finger' keys, and the interpipe dictionary contains 'trial_idx'.
		It uses `neuraldecoding.utils.data_split_trial` to perform the split.
		Args:
			data (dict): Input data dictionary containing 'neural' and 'finger' keys.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (dict): A dictionary containing:
				- 'neural_train': Training neural data
				- 'neural_test': Testing neural data
				- 'finger_train': Training finger data
				- 'finger_test': Testing finger data
				if split_ratio is a tuple, also includes:
				- 'neural_val': Validation neural data
				- 'finger_val': Validation finger data
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if 'trial_idx' not in interpipe:
			raise ValueError("DataSplitBlock requires 'trial_idx' in interpipe from other wrappers (Dict2DataBlock).")

		split_data = neuraldecoding.utils.data_split_trial(data['neural'], 
														   data['finger'], 
														   interpipe['trial_idx'], 
														   split_ratio=self.split_ratio, 
														   seed=self.split_seed)
		
		data, test_start_idx = split_data
		
		interpipe['test_start_idx'] = test_start_idx

		return data, interpipe

class Dict2FingerDatasetBlock(DataFormattingBlock):
	"""
	Converts a dictionary to a FingerDataset or FingerDatasetCustom class.
	"""
	def __init__(self, otherdatakeys_list=None, otherinterpipekeys_list=None):
		"""
		Initializes the Dict2FingerDatasetBlock. Creates FingerDataset objects for train, validation (if included) and test. 
		FingerDatasetCustom allows for additional data/interpipe keys to be included in the dataset. otherdatakeys_list and
		otherinterpipekeys_list can be list of lists. Ensure that otherdatakeys_list and otherinterpipekeys_list have settings
		in the order of [train, valid, test] or [train, test] if no validation set is used.
		e.g. ,
			otherdatakeys_list = ['trial_lengths_train', None], otherinterpipekeys_list = [['leadup'], None]
		Args:
			otherdatakeys_list (str, list, optional): Additional locations in the data dictionary to include. Must be length 2 if 
				only train and test datasets are used, or length 3 if train, valid, and test datasets are used, or None.
			otherinterpipekeys_list (str, list, optional): Additional keys in the interpipe dictionary to include. Must be length 2 if
				only train and test datasets are used, or length 3 if train, valid, and test datasets are used, or None.
		"""
		super().__init__()

		self.otherdatakeys_list = otherdatakeys_list
		self.otherinterpipekeys_list = otherinterpipekeys_list
	
	def transform(self, data, interpipe):
		"""
		Transform the data from a dictionary to FingerDataset or FingerDatasetCustom class.
		Args:
			data (dict): Input data dictionary containing the data and relevant information.
			interpipe (dict): A inter-pipeline bus for one-way sharing data between blocks within the preprocess_pipeline call.
		Returns:
			data_out (tuple): A tuple of FingerDataset or FingerDatasetCustom instances in order of [train, valid, test] or [train, test].
			interpipe (dict): The interpipe dictionary remains unchanged.
		"""
		if 'neural_val' in data:
			self.xkey = ['neural_train', 'neural_val', 'neural_test']
			self.ykey = ['finger_train', 'finger_val', 'finger_test']
			self.savekeys = ['train', 'valid', 'test']
		else:
			self.xkey = ['neural_train', 'neural_test']
			self.ykey = ['finger_train', 'finger_test']
			self.savekeys = ['train', 'test']

		if self.otherdatakeys_list is None:
			self.otherdatakeys_list = [None] * len(self.xkey)
		if self.otherinterpipekeys_list is None:
			self.otherinterpipekeys_list = [None] * len(self.xkey)
		
		if len(self.otherdatakeys_list) != len(self.xkey) or len(self.otherinterpipekeys_list) != len(self.xkey):
			raise ValueError("otherdatakeys_list and otherinterpipekeys_list size mismatch.")

		datasets = ()
		for xkey, ykey, savekey, otherdatakeys, otherinterpipekeys in zip(self.xkey, self.ykey, self.savekeys, self.otherdatakeys_list, self.otherinterpipekeys_list):
			if otherdatakeys is None and otherinterpipekeys is None:
				dataset = neuraldecoding.utils.datasets.FingerDataset(data, xkey, ykey)
			else:
				dataset = neuraldecoding.utils.datasets.FingerDatasetCustom(data, interpipe, xkey, ykey, otherdatakeys, otherinterpipekeys)
			datasets += (dataset,)
		
		return datasets, interpipe

class Dataset2DictBlock(DataFormattingBlock):
	"""
	Converts a dictionary (from load_one_nwb) to neural and finger data in dictionary format.
	Add 'trial_idx' to interpipe.
	"""
	def __init__(self, neural_nwb_loc, behavior_nwb_loc, time_nwb_loc, apply_trial_filtering = True):
		"""
		Initializes the Dict2DataDictBlock.
		Args:
			neural_type (str): Type of neural data to extract from the dictionary. Default is "sbp". Can be "sbp" or "tcfr"
		"""
		self.neural_nwb_loc = neural_nwb_loc
		self.behavior_nwb_loc = behavior_nwb_loc
		self.time_nwb_loc = time_nwb_loc
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
		
		neural, finger = resolve_path(data.dataset, self.neural_nwb_loc)[:], resolve_path(data.dataset, self.behavior_nwb_loc)[:]
		time_stamps = resolve_path(data.dataset, self.time_nwb_loc)[:]
		
		try:
			assert(len(neural) == len(time_stamps) == len(finger))
		except:
			ValueError("Dimension mismatch")
	
		if self.apply_trial_filtering:
			UserWarning("Trial Filtering coming soon to a dataset near you")
		
		interpipe['time_stamps'] = time_stamps

		data_out = {'neural': neural, 'finger': finger}
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
		trial_ts_train = interpipe['trial_ts'][:interpipe['test_start_idx']]
		data['neural_train'], data['finger_train'], trial_lengths_train = \
			neuraldecoding.utils.add_trial_history(data['neural_train'], data['finger_train'], trial_ts_train, self.leadup)
		data['trial_lengths_train'] = trial_lengths_train
		interpipe['leadup'] = self.leadup

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
	def __init__(self, location, feature_extractor_config):
		super().__init__()
		self.location = location
		self.feature_extractor = FeatureExtractor(feature_extractor_config)

	def transform(self, data, interpipe):
		for loc in self.location:
			if loc not in data:
				raise ValueError(f"Location '{loc}' not found in data dictionary.")
			data[loc] = self.feature_extractor.extract_binned_features(data=data[loc], timestamps_ms=interpipe.get('time_stamps', None))
		return data, interpipe
	
class LabelModificationBlock(DataProcessingBlock):
	"""
	A block to add label modifications to training and/or testing data.
	"""

	def __init__(self, modifications, param_dict, location):
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
			leadup (int): The length of the history to be added before the first bin of each trial. Default is 10.
			location (str or list): where to apply the modifications
		"""
		self.modifications = modifications
		self.param_dict = param_dict
		self.location = location
	
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
		if isinstance(self.location, str):
			self.location = [self.location]

		for loc in self.location:
			data[loc] = neuraldecoding.utils.label_mods.apply_modifications(self.modifications, data[loc], interpipe['trial_ts'], self.param_dict)
		
		return data, interpipe
	
