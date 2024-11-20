import os
import numpy as np
import pandas as pd
from pynwb import NWBFile
from hdmf.backends.hdf5.h5_utils import H5DataIO
from neuroconv.tools.hdmf import SliceableDataChunkIterator
from datetime import datetime
from dateutil.tz import tzlocal

from .utils_dataset import *
from .utils_server import *
from ..utils import *

'''
    To discuss:
        - is ok to set the data_type in the constructor of the Dataset? Or should it be set in the load_data method?
           (the question regards if the data_type is fixed for all the dataset or we leave the option of loading differe data types in the same dataset)
'''
class Dataset:
    def __init__(self, is_monkey=True, data_type='utah', verbose=False):
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

        # TODO: add the check if the dataset was already created: in that case just load it
        # loading each run
        for run in runs:   
            if run not in self.runs:
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
        pass

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
        data_frame = read_xpc_data(contents, run_path, num_channels=self.num_channels, verbose=self.verbose)

        # TODO: consider to change this part to handle the data columns initializaition predefined and indipendent of the data content
        time_series_dict = self.__initialize_nwb_columns(run, data_frame)

        self.__add_run_data(run, data_frame, time_series_dict)

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
        
        return nwb_file
    
    def __initialize_nwb_columns(self, run, data_frame):
        """
        Initialize the columns of the NWB file based on the data frame. Returns the time_series dictionary

        Inputs:
            run: int
                Run number
            data_frame: pandas.DataFrame
                Data frame with the data from the server

        Outputs:
            time_series_dict: dict
                Dictionary of the time series data initialized
        """
        # Dynamically create trial column and create the time_series dictionary
        time_series_dict = dict()

        for key in data_frame.keys():
            if key == 'Channel' or key == 'ExperimentTime':
                continue  # Channel (spike times) and ExperimentTime are handled as special cases
            elif not is_collection(data_frame[key][0]) or len(data_frame[key].iloc[-1]) <= 5:            
                self.nwb_files[run].add_trial_column(name=key, description=key)
            else:
                time_series_dict[key] = []

        return time_series_dict

    def __add_run_data(self, run, data_frame, time_series_dict):
        spike_times = [[[] for i in range(self.num_channels)] for j in range(1)]
        obs_intervals = []
        experiment_time = []

        # adding general run data
        # TODO for loop may be able to be simplified by changing ExperimentTime format in ReadData. Might be able to use [:] instead of going through each trial on next line
        for idx in range(len(data_frame)):
            # Dynamically create data in trial columns
            nwb_trial_dict = dict()

            for key in data_frame.keys():
                if key == 'Channel':
                    obs_intervals.append([data_frame['ExperimentTime'][idx][0][0], data_frame['ExperimentTime'][idx][-1][0]])\
                    
                    for ch_id in range(self.num_channels):
                        val_to_add = data_frame['Channel'][idx][ch_id]['SpikeTimes']
                        if isinstance(val_to_add, np.ndarray):
                            spike_times[0][ch_id].append(val_to_add.tolist())
                        elif isinstance(val_to_add, (np.int32, np.int64, np.float64)):
                            spike_times[0][ch_id].append([val_to_add.item()])
                        elif isinstance(val_to_add, list):
                            continue
                        else:
                            raise Warning(f"Couldn't find type: {type(val_to_add)}")
                elif key == 'ExperimentTime':
                    nwb_trial_dict['start_time'] = data_frame['ExperimentTime'][idx][0][0]
                    nwb_trial_dict['stop_time'] = data_frame['ExperimentTime'][idx][-1][0]

                    experiment_time.extend(x for xs in data_frame['ExperimentTime'][idx].tolist() for x in xs)
                #TODO Use zScript.txt to determine timeseries and trial data as opposed to length of data
                elif not is_collection(data_frame[key][0]) or len(data_frame[key].iloc[-1]) <= 5:
                    # create trial dictionary for later use
                    nwb_trial_dict[key] = data_frame[key][idx]                
                else:
                    # create timeseries dictionary for later user
                    time_series_dict[key].extend(data_frame[key][idx].tolist())

            self.nwb_files[run].add_trial(**nwb_trial_dict)

        # adding the time series data
        first_key = True

        for key in time_series_dict.keys():
            if first_key:
                first_key_time_series = TimeSeries(
                    name = key,
                    data = H5DataIO(
                        data = SliceableDataChunkIterator(
                            data = np.array(time_series_dict[key])), 
                            compression = self.nwb_compression['type'], 
                            compression_opts = self.nwb_compression['options']),
                    unit = '',
                    timestamps = H5DataIO(
                        data = SliceableDataChunkIterator(
                            data = np.array(experiment_time)), 
                            compression = self.nwb_compression['type'], 
                            compression_opts = self.nwb_compression['options']),
                    conversion=1.0
                    )
                self.nwb_files[run].add_acquisition(first_key_time_series)
                first_key = False
            else:
                self.nwb_files[run].add_acquisition(TimeSeries(
                    name = key,
                    data = H5DataIO(
                        data=SliceableDataChunkIterator(
                            data=np.array(time_series_dict[key])), 
                            compression=self.nwb_compression['type'], 
                            compression_opts=self.nwb_compression['options']),
                    unit = '',
                    timestamps = first_key_time_series,
                    conversion = 1.0
                ))

        # adding unit data
        for ch_id in range(self.num_channels):
            self.nwb_files[run].add_unit(
                spike_times = [float(item) for row in spike_times[0][ch_id] for item in row],
                electrodes = [ch_id],
                obs_intervals=obs_intervals)
            
        if self.verbose:
            print(f"Run {run} loaded successfully")

    # TODO
    def __str__(self):
        pass