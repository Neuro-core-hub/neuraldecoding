import os
import numpy as np
import pandas as pd
import itertools
from pynwb import NWBFile
from hdmf.backends.hdf5.h5_utils import H5DataIO
from neuroconv.tools.hdmf import SliceableDataChunkIterator
from datetime import datetime
from dateutil.tz import tzlocal

from .utils_dataset import *
from .utils_server import *
from ..utils import *

class Dataset:
    def __init__(self, is_monkey=True, data_type='utah', verbose=True):
        """
        Inputs:
            is_monkey: bool - Default: True
                Flag to specify if the dataset is from monkey or human recordings
            type: str - Default: 'utah'
                Possible values: 'utah', 'emg'
        """
        self.is_monkey = is_monkey
        self.verbose = verbose

        self.institution, self.lab = get_creator_details()
        self.device, self.electrode_group = get_device_and_electrode_group(data_type)
        self.server_dir = get_server_data_path(is_monkey=is_monkey)

        self.runs = []
        self.nwb_modules = {}
        self.nwb_files = {}

        # setting the default NWB compression options - if needed to changed they can be moved to a config file
        self.nwb_compression = {
            'type': 'gzip',
            'options': 4
        }

    def set_server_directory(self, server_dir):
        """
        Set manually the server route directory (if different from the default for windows/linux systems)

        Inputs:
            server_dir: str
                Server directory path (e.g. 'F:' if the server is mounted on the F drive)
        """
        self.server_dir = get_server_data_path(is_monkey=self.is_monkey, custom_server_folder=server_dir)
    
    # TODO: add the functionality that checks if the NWB file exists and if yes it just loads it
    def load_data(self, subject_name=None, date=None, data_path=None, runs=None, num_channels=96):
        """
        Load data from server file path or from specific subject-date-runs

        Inputs:
            subject_name: str
                Subject name
            dates: str
                Date of the recording or None if loading from data_paths directly
            data_path: str
                Folder path to load the data from (assuming it is a folder with the runs folders inside) or None if loading using subject-date format
            runs: list of int
                List of runs from which loading the data
            num_channels: int - Default: 96
                Number of channels
        """

        if (subject_name is None or date is None) and data_path is None:
            raise ValueError("Either ('subject' and 'date') or 'file_path' must be specified")

        if runs is None:
            raise ValueError("Parameter 'runs' must be specified")
        
        self.num_channels = num_channels

        # setting the dataset details (data_path and subject object)
        self.__set_dataset_details(subject_name, date, data_path)

        if self.verbose:
            print(f"Loading data for {self.subject.description} from {self.data_path}")

        # TODO: add the check if the dataset was already created: in that case just load it
        # loading each run
        for run in runs:   
            if run not in self.runs:
                if self.verbose:
                    print(f" + Loading run {run}")

                self.runs.append(run)

                # initialize the NWB file
                self.nwb_files[run] = self.__initialize_nwb_file(run)  

                # load the data from the server into the nwb_file
                self.__load_data_server(run)
            else:
                raise Warning(f"Run {run} already loaded")

    def sync_data_cpd(file_path, run, start_run, params=[]):
        pass

    def sync_data_cer(file_path, run, start_run, params=[]):
        pass

    def processing_data(self, fields, params=[]):
        pass

    def extract_features(self, fields, params=[]):
        """
        Extract the features from the neural data
        
        Inputs:
            fields: list of str
                List of fields to extract the features from
            params: list of dict
                List of parameters for the feature extraction
        Outputs:
            features: dictionary containing the numpy matrix for each feature
        """
        # Inputs and params validation and initialization
        assert len(fields) > 0, "At least one field must be specified for the feature extraction"
        assert 'bin_size' in params, "Parameter 'bin_size' must be specified for the feature extraction"

        bin_size = params['bin_size']
        behav_lag = params['behav_lag'] if 'behav_lag' in params else None
        overlap = params['overlap'] if 'overlap' in params else 0
        remove_first_trial = params['remove_first_trial'] if 'remove_first_trial' in params else False

        if not np.isscalar(bin_size) or not (type(bin_size) == int or type(bin_size) == float):
            raise Exception('Parameter bin_size must be a numeric scalar!')

        if behav_lag and (not np.isscalar(behav_lag) or not (type(behav_lag) == int or type(behav_lag) == float)):
            raise Exception('Parameter behav_lag must be a numeric scalar!')

        if overlap and (not (type(overlap) == int)):
            raise Exception('Parameter overlap must be an integer number!')

        try:
            remove_first_trial = bool(remove_first_trial)
        except:
            raise Exception('Parameter remove_first_trial must be a boolean!')

        # variables initialization
        features = {field: None for field in fields}

        if behav_lag:
            behav_lag_samples = int((np.ceil(behav_lag) / bin_size))

        # looping through the runs for extracting the features
        for run in self.runs:
            # retrieving trials info and data needed for the features extraction
            trials = self.nwb_files[run].trials.to_dataframe()
            timestamps = self.nwb_files[run].processing['neural_data'].get_data_interface('neural_features').timestamps[:] # TODO: there might be a better way of doing this

            if 'sbp' in fields:
                neural_data = self.nwb_files[run].processing['neural_data'].get_data_interface('neural_features').data[:]
                sample_width = self.nwb_files[run].acquisition['SampleWidth'].data[:] # TODO: change this when the normalization of the names will be done

            if 'fingers_kinematics' in fields:
                fingers_pos_data = self.nwb_files[run].processing["behavior"].get_data_interface('fingers_position').data[:]
                num_fingers = fingers_pos_data.shape[1]

            # trials filtering
            if remove_first_trial:
                trials = trials.iloc[1:]

            if 'trials_filter' in params:
                filter_conditions = params['trials_filter']

                # TODO: once the features naming will be normalized replace the 'if statements' with a forloop over the filter_conditions
                if 'blank_trial' in filter_conditions:
                    trials = trials[trials['BlankTrial'] == filter_conditions['blank_trial']]
                if 'trial_success' in filter_conditions: 
                    trials = trials[trials['TrialSuccess'] == filter_conditions['trial_success']]

            num_trials = len(trials)
            start_times = trials['start_time'].values
            stop_times = trials['stop_time'].values
            trial_numbers = trials['TrialNumber'].values   # TODO: standardize the name of TrialNumber to trial_number

            # finding the non-consecutive runs of trials - used later for consecutive features extraction
            trials_runs = np.where(np.diff(trial_numbers) != 1)[0] # non-consecutive run are delimited where the trial-num difference is not 1           
            trials_runs = np.concatenate([[-1], trials_runs, [num_trials-1]]) # adding first and last trial index

            for run_id in range(len(trials_runs) - 1):
                start_trial = trials_runs[run_id] + 1
                stop_trial = trials_runs[run_id + 1]

                start_time = start_times[start_trial]
                stop_time = stop_times[stop_trial]
                start_time_index = np.argmax(timestamps >= start_time)
                stop_time_index = np.argmax(timestamps >= stop_time) + 1
                run_times = timestamps[start_time_index:stop_time_index]

                # timing of bins computation
                desidered_last_bin_time = stop_time - bin_size
                num_bins = int(np.fix((desidered_last_bin_time - start_time) / bin_size))
                last_bin_time = start_time + num_bins * bin_size

                bins_start = np.linspace(start_time, last_bin_time, num_bins + 1)
                bins_stop = bins_start + bin_size - 1

                # digitize bins times values into bins defined by the midpoints
                mean_offsets = (run_times[1:] + run_times[:-1]) / 2
                bins_start_digit = np.digitize(bins_start, bins=mean_offsets) 
                bins_stop_digit = np.digitize(bins_stop, bins=mean_offsets)

                if behav_lag:  
                    bins_start_digit_lag = bins_start_digit - behav_lag_samples
                    bins_stop_digit_lag = bins_stop_digit - behav_lag_samples

                    # Remove values that are less than 0 (and keep the rest that are greater than or equal to 0)
                    bins_start_digit_lag = bins_start_digit_lag[bins_start_digit_lag >= 0]
                    bins_stop_digit_lag = bins_stop_digit_lag[bins_stop_digit_lag >= 0]

                # extracting the features for each field
                for field in fields:
                    if field == 'sbp' or field == 'mav': # Spiking Band Power or Mean Absolute Value - TODO: maybe change the name of these two fields                    
                        sbp_run_data = neural_data[start_time_index:stop_time_index, :]
                        sample_width_run = sample_width[start_time_index:stop_time_index]

                        bins_start = bins_start_digit_lag if behav_lag else bins_start_digit
                        bins_stop = bins_stop_digit_lag if behav_lag else bins_stop_digit
  
                        run_features = calc_bins_sbp(sbp_run_data, sample_width_run, bins_start, bins_stop)
                        
                    elif field == 'fingers_kinematics': # Fingers kinematics (position, velocity, acceleration)
                        fingers_pos_run_data = fingers_pos_data[start_time_index:stop_time_index, :]

                        bins_start = bins_start_digit - bins_start_digit[0]
                        bins_stop = bins_stop_digit - bins_start_digit[0]

                        run_features = calc_bins_fingers_kinematics(fingers_pos_run_data, bins_start, bins_stop, num_fingers)

                        # Remove samples based on lag_ms
                        if behav_lag:
                            run_features = run_features[behav_lag_samples:]
                    else:
                        raise ValueError(f"Field '{field}' not supported for feature extraction yet")

                    if features[field] is None:
                        features[field] = run_features
                    else:
                        features[field] = np.concatenate((features[field], run_features), axis=0)                    
               
        return features

    def save_data(self, fields):
        pass

    # load existing NWB file - already generated with load_data before
    def __load_data_existing(self, file_path):
        pass

    def __set_dataset_details(self, subject_name=None, date=None, data_path=None):
        """
        Set the data path and the subject object for a specific subject and date

        Inputs:
            subject_name: str
                Subject name
            date: str
                Date of the recording
            data_path: str
                Folder path to load the data from (assuming it is a folder with the runs folders inside) or None if loading using subject-date format
        """
        if data_path is None:
            data_path = os.path.join(self.server_dir, subject_name, date)

        # normalizing the path to be OS independent
        self.data_path = os.path.normpath(data_path)

        # retrieving the subject name from the data_path if not directly provided
        if subject_name is None:
            subject_name = self.data_path.split(os.sep)[-2]

        # setting the subject object
        self.subject = get_subject(subject_name)

        # setting the creation time
        self.creation_time = get_creation_path_time(self.data_path)

        # setting the experimenter and notes content
        self.experimenter, self.notes_content = get_server_notes_details(self.data_path)

    def __initialize_nwb_file(self, run):
        """
        Initialize the NWB file

        Inputs:
            run: int
                Run number

        Outputs:
            nwb_file: pynwb.file.NWBFile
                NWB file object
        """
        
        nwb_file = NWBFile(
            session_description = f"{self.data_path} | Run-{run:03d}",
            identifier = f"{self.data_path}/Run-{run:03d}",
            session_start_time = datetime.fromtimestamp(self.creation_time, tz=tzlocal()),
            file_create_date = datetime.now(tzlocal()),
            experimenter = self.experimenter,
            institution = self.institution,
            notes = self.notes_content,
            lab = self.lab,
            subject = self.subject,
            devices = [self.device],
            electrode_groups = [self.electrode_group])
        
        for _ in range(self.num_channels):
            nwb_file.add_electrode(group=self.electrode_group, location=self.electrode_group.location)

        return nwb_file

    def __load_data_server(self, run):
        """
        Load the data from the server and populate the nwb_file
        
        Inputs:
            run: int
                Run number
        """
        run_path = os.path.join(self.data_path, f'Run-{run:03d}')

        # loading the zScript file content
        contents = load_z_script(run_path)

        # reading the xpc data
        if self.verbose:
            print(f"   - Reading server data")

        data_dict = read_xpc_data(contents, run_path, num_channels=self.num_channels, verbose=self.verbose)

        # initializing the nwb modules and columns
        self.__initialize_nwb_modules(run)
        time_series_dict = self.__initialize_nwb_columns(run, data_dict) # TODO: consider to change this part to handle the data columns initializaition predefined and indipendent of the data content

        # populate the nwb file with the run data
        self.__add_run_data(run, data_dict, time_series_dict)
    
    def __initialize_nwb_modules(self, run):
        """
        Initialize the NWB modules for the run NWB file

        Inputs:
            run: int
                Run number
        """
        behavior_module = self.nwb_files[run].create_processing_module(
            name="behavior", description="Raw behavioral data"
        )

        neural_data_module = self.nwb_files[run].create_processing_module(
            name="neural_data", description="Neural data"
        )

        self.nwb_modules[run] = {
            'behavior': behavior_module,
            'neural_data': neural_data_module
        }

    def __initialize_nwb_columns(self, run, data_dict):
        """
        Initialize the columns of the NWB file based on the dictionary keys. Returns the time_series key names dictionary

        Inputs:
            run: int
                Run number
            data_dict: list of dictionaries
                Dictionary with the data from the server

        Outputs:
            time_series_dict: dict
                Dictionary of the time series data initialized
        """

        # retrieve the behavior variable name from the config file
        self.behavior_var_name = get_dataset_variable_name('behavior_data') 
        self.neural_var_name = get_dataset_variable_name('neural_data')

        # Dynamically create trial column and create the time_series dictionary
        time_series_dict = dict()

        num_trials = len(data_dict)
        num_times = len(data_dict[-1]['ExperimentTime']) # num of times for last trial - used to understand if a variable is a time-series
        num_total_times = sum(len(data_dict[trl_idx]['ExperimentTime']) for trl_idx in range(num_trials))
        
        for key in data_dict[0].keys():
            if key == 'ExperimentTime':
                continue  # ExperimentTime are handled as special cases
            elif key == self.behavior_var_name or key == self.neural_var_name:
                continue # Behavior data and SBP are handled as special case (stored in the respective modules)                
            elif not is_collection(data_dict[0][key]) or len(data_dict[-1][key]) != num_times:            
                self.nwb_files[run].add_trial_column(name=key, description=key)
            else:
                time_series_dict[key] = np.empty((num_total_times, data_dict[0][key].shape[1]), dtype=data_dict[0][key].dtype)

        return time_series_dict

    def __add_run_data(self, run, data_dict, time_series_dict):
        """
        Add the data from the server data dictionary to the NWB file
        Inputs:
            run: int
            data_dict: list of dictionaries
            time_series_dict: dict
        """

        if self.verbose:
            print(f"   - Converting data to NWB file")

        num_trials = len(data_dict)
        num_behavior_vars = data_dict[0][self.behavior_var_name].shape[1]
        experiment_time = []

        total_times = sum(len(data_dict[trl_idx]['ExperimentTime']) for trl_idx in range(num_trials))

        # preallocate time-series data arrays
        times = np.empty((total_times,), dtype=data_dict[0]['ExperimentTime'].dtype)
        neural_data = np.empty((total_times,self.num_channels), dtype=data_dict[0][self.neural_var_name].dtype)
        behavior_data = np.empty((total_times,num_behavior_vars), dtype=data_dict[0][self.behavior_var_name].dtype)

        # looping through the trials and populating the times-series data
        start_idx = 0

        for trl_idx in range(num_trials):
            times_trial_data = np.squeeze(data_dict[trl_idx]['ExperimentTime'])

            end_idx = start_idx + len(times_trial_data)

            times[start_idx:end_idx] = times_trial_data
            neural_data[start_idx:end_idx] = data_dict[trl_idx][self.neural_var_name]
            behavior_data[start_idx:end_idx] = data_dict[trl_idx][self.behavior_var_name]

            # add all the other time-series data
            for key in time_series_dict.keys():
                time_series_dict[key][start_idx:end_idx] = data_dict[trl_idx][key]

            start_idx = end_idx

        # adding behavior data
        fingers_position_ts = TimeSeries(
            name="fingers_position",
            data=H5DataIO(
                behavior_data,                
                compression=self.nwb_compression['type'], 
                compression_opts=self.nwb_compression['options']
            ),
            unit="flexion units",  # From 0 to 1
            timestamps=times,
            description="Fingers flexion position, from fully extended (0) to fully flexed (1)",
            comments="Raw position data"
        )

        self.nwb_modules[run]['behavior'].add_data_interface(fingers_position_ts)
            
        # adding neural data
        neural_data_ts = TimeSeries(
            name="neural_features",
            data=H5DataIO(
                neural_data,
                compression=self.nwb_compression['type'], 
                compression_opts=self.nwb_compression['options']
            ),
            unit="mV",
            timestamps=times,
            description="Neural data features across time",
            conversion=1.0,
            comments="Neural data features, for chesteklab it's the Spiking Band Power (SBP) - i.e. the Mean Absolute Value, for each 1ms bin"
        )

        self.nwb_modules[run]['neural_data'].add_data_interface(neural_data_ts)

        # adding all the other time-series data as acquisition data
        for key in time_series_dict.keys():
            self.nwb_files[run].add_acquisition(
                TimeSeries(
                    name=key,
                    unit="",
                    data=H5DataIO(
                        time_series_dict[key], 
                        compression=self.nwb_compression['type'], 
                        compression_opts=self.nwb_compression['options']
                    ),
                    timestamps=times,
                    description=f"{key}",
                    conversion=1.0
                )
            )

        # looping through the trials for adding the trials and all the other non-timeseries data to the NWB file
        
        # base structure of each trial
        nwb_trial_dict = dict()

        # add the trials to the NWB file
        for trl_idx in range(num_trials):
            # trial times
            nwb_trial_dict['start_time'] = data_dict[trl_idx]['ExperimentTime'][0][0]
            nwb_trial_dict['stop_time'] = data_dict[trl_idx]['ExperimentTime'][-1][0]

            # looping through the data frame keys for dynamically adding them (TODO: the keys should be defined in a config file)
            for key in data_dict[trl_idx].keys():       
                if key == 'ExperimentTime':
                    pass # ExperimentTime is handled as a special case    
                elif key == self.behavior_var_name or key == self.neural_var_name:
                    pass
                elif key in time_series_dict.keys():
                    pass
                else: # non-time series data
                    if key == 'TrialNumber':
                        data_dict[trl_idx][key] -= 1  # subtract 1 since indexes in the matlab version start from 1
                    
                    nwb_trial_dict[key] = data_dict[trl_idx][key]

            self.nwb_files[run].add_trial(**nwb_trial_dict)

        if self.verbose:
            print(f"   - Run {run} loaded successfully")

    # TODO - it should return a representation string of the days/runs loaded in the dataset
    def __str__(self):
        pass

    # TODO - it should return the total number of runs
    def __len__(self):
        pass

    # TODO
    def __add__(self):
        pass

    # TODO
    def __sub__(self):
        pass

''' Useful instructions for nwb

- get the behavior module: nwbfile.processing["behavior"]
    + get timeseries from behavior module: nwbfile.processing["behavior"].get_data_interface('fingers_position')
- removing a timeseries column: nwbfile.acquisition.pop("timeseries-name")

'''