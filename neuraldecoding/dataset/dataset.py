import os
import numpy as np
import pandas as pd
import yaml
from pynwb import NWBHDF5IO, NWBFile
from datetime import datetime
from dateutil.tz import tzlocal

from .utils_dataset import *
from .utils_server import *
from ..utils import *

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

'''
    To discuss:
        - is ok to set the data_type in the constructor of the Dataset? Or should it be set in the load_data method?
           (the question regards if the data_type is fixed for all the dataset or we leave the option of loading differe data types in the same dataset)
'''

class Dataset:
    def __init__(self, is_monkey=True, data_type='utah'):
        """
        Inputs:
            is_monkey: bool - Default: True
                Flag to specify if the dataset is from monkey or human recordings
            type: str - Default: 'utah'
                Possible values: 'utah', 'emg'
        """
        self.is_monkey = is_monkey
        self.institution, self.lab = get_creator_details()
        self.device, self.electrode_group = get_device_and_electrode_group(data_type)
        self.server_dir = get_server_data_path(is_monkey=is_monkey)

    def set_server_directory(self, server_dir):
        """
        Set manually the server route directory (if different from the default for windows/linux systems)

        Inputs:
            server_dir: str
                Server directory path (e.g. 'F:' if the server is mounted on the F drive)
        """
        self.server_dir = get_server_data_path(is_monkey=self.is_monkey, custom_server_folder=server_dir)
    
    # TODO: add the functionality that checks if the NWB file exists and loads it if it does
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

        data_path, subject = self.__get_directory_details(subject_name, date, data_path)

        # loading each run
        for run in runs:
            # TODO: add the check if the dataset was already loaded (it might be a mistake by the user to call load twice on the same data)            
            self.__load_data_server(subject, data_path, run)

    # load existing NWB file - already generated with load_data before
    def load_data_existing(self, file_path):
        pass

    def sync_data_cpd(file_path, run, start_run, params=[]):
        pass

    def sync_data_cer(file_path, run, start_run, params=[]):
        pass

    def processing_data(self, fields, params=[]):
        pass

    def exctract_features(self, fields, params=[]):
        pass

    def save_data(self, fields):
        pass



    def __get_directory_details(self, subject_name=None, date=None, data_path=None):
        """
        Set the data path and the subject object for a specific subject and date

        Inputs:
            subject_name: str
                Subject name
            date: str
                Date of the recording
            data_path: str
                Folder path to load the data from (assuming it is a folder with the runs folders inside) or None if loading using subject-date format

        Outputs:
            data_path: str
                Folder path to load the data from
            subject: pynwb.file.Subject
                Subject object
        """
        if data_path is None:
            data_path = os.path.join(self.server_dir, subject_name, date)

        data_path = os.path.normpath(data_path)  # this is system independent splitting of path

        # retrieving the subject
        if subject_name is None:
            folders = os.path.split(os.sep) # this is system independent splitting of path
            subject_name = folders[-2]

        subject = get_subject(subject_name)

        return data_path, subject

    def __load_data_server(self, subject, data_path, run):
        run_path = os.path.join(data_path, f'Run-{run:03d}')

        # loading the zScript file content
        contents = load_z_script(run_path)

        # reading the xpc data
        # z = read_xpc_data(contents, run_path, num_channels=self.num_channels, verbose=True)

        nwb_file = self.__initialize_nwb_file(subject, data_path)

    def __initialize_nwb_file(self, subject, data_path):
        creation_time = get_creation_path_time(data_path)
        experimenter, notes_content = get_server_notes_details(data_path)
        
        nwb_file = NWBFile(
            session_description=data_path,
            identifier=data_path,
            session_start_time=datetime.fromtimestamp(creation_time),
            file_create_date=datetime.now(tzlocal()),
            experimenter=experimenter,
            institution=self.institution,
            notes=notes_content,
            lab=self.lab,
            subject=subject,
            devices=[self.device],
            electrode_groups=[self.electrode_group])
        
        return nwb_file

    # TODO
    def __str__(self):
        pass