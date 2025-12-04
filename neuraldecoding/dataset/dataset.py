import os
import numpy as np
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.ecephys import ElectricalSeries
from datetime import datetime
from dateutil.tz import tzlocal
from . import zstruct_loader

from omegaconf import DictConfig


class Dataset:
    def __init__(self, cfg: DictConfig, verbose=True):
        """
        Initialize a Dataset object for neural data processing.

        Parameters
        ----------
        cfg: Configuration with the following keys (usually saved as a yaml file)
            dataset_type: str, specifies the type of dataset being loaded (and will use the corresponding loading functions)
            autoload: bool, specifies if the dataset should be automatically loaded when the dataset object is instantiated
            save_path: str, filepath for where data should be saved (otherwise will send to a default location)
            dataset_parameters: config block, contains all the parameters needed for the dataset_type specified above.
        """

        self.cfg: DictConfig = cfg
        self.dataset_parameters: DictConfig = self.cfg.dataset_parameters
        self.verbose: bool = verbose
        # Initialize empty NWB file
        self.dataset: NWBFile = NWBFile(
            session_description="",
            identifier="",
            session_start_time=datetime.now(tzlocal()),
        )
        self.io = None # Added file handle storage

        # if self.cfg.dataset_type == "zstruct":
        #     self._initialize_zstruct()
        # elif self.cfg.dataset_type == "nwb":
        #     self._initialize_nwb()
        # else:
        #     raise NotImplementedError(f"Unimplemented dataset type: {self.cfg.dataset_type}")
        
        if self.cfg.autoload:
            self.load_data()

    def load_data(self):
        """
        Load data based on the dataset type specified in configuration.
        Delegates to the appropriate loading method based on dataset type.

        Raises
        ------
        NotImplementedError
            If the dataset type is not supported
        """
        if self.cfg.dataset_type == "zstruct":
            self._load_data_zstruct()
        elif self.cfg.dataset_type == "nwb":
            self._load_data_nwb()
        elif self.cfg.dataset_type == "multi-nwb":
            self._load_data_multi_nwb()
        else:
            raise NotImplementedError(f"Unimplemented dataset type: {self.cfg.dataset_type}")
    
    def _load_data_multi_nwb(self):
        """
        Load data from multiple NWB files
        Combines them into a new NWB file using the first file as the template and adding on top
        """
        self.io = NWBHDF5IO(self.cfg.dataset_parameters.nwb_files[0], mode="r")
        self.dataset = self.io.read()
        for nwb_file in self.cfg.dataset_parameters.nwb_files[1:]:
            # Read the NWB file
            with NWBHDF5IO(nwb_file, mode="r") as io:
                nwbdata = io.read()
                if self.cfg.dataset_parameters.patterns_add_mode == "new_patterns":
                    # Pop original timeseries
                    # Add new patterns
                    old_patterns = self.dataset.acquisition.pop("patterns")
                    new_patterns = old_patterns.data[:]
                    
                    if new_patterns.ndim == 1:
                        new_patterns = new_patterns.reshape(-1, 1)
                    # Get patterns from current nwb file
                    additional_patterns = nwbdata.acquisition["patterns"].data[:]
                    if additional_patterns.ndim == 1:
                        additional_patterns = additional_patterns.reshape(-1, 1)
                    additional_patterns = np.hstack([np.zeros((additional_patterns.shape[0], new_patterns.shape[1])), additional_patterns])
                    new_patterns = np.hstack([new_patterns, np.zeros((new_patterns.shape[0], nwbdata.acquisition["patterns"].data.shape[1]))])
                    # Vstack new and additional patterns
                    new_patterns = np.vstack([new_patterns, additional_patterns])
                    # Modify the units
                    new_patterns_unit = eval(old_patterns.unit) + eval(nwbdata.acquisition["patterns"].unit)
                    new_patterns_unit = str(new_patterns_unit)
                    new_patterns_timestamps = np.concatenate([old_patterns.timestamps[:], nwbdata.acquisition["patterns"].timestamps[:]])
                    self.dataset.add_acquisition(TimeSeries(name="patterns", data=new_patterns, timestamps=new_patterns_timestamps, unit=new_patterns_unit, description=old_patterns.description))
                if self.cfg.dataset_parameters.neural_add_mode == "simple":
                    # Pop original timeseries
                    # Add new neural
                    old_neural = self.dataset.acquisition.pop("neural")
                    new_neural = old_neural.data[:]
                    
                    # Get neural from current nwb file
                    additional_neural = nwbdata.acquisition["neural"].data[:]
                    new_neural = np.vstack([new_neural, additional_neural])
                    # Modify timestamps
                    new_neural_timestamps = np.concatenate([old_neural.timestamps[:], nwbdata.acquisition["neural"].timestamps[:]])
                    self.dataset.add_acquisition(ElectricalSeries(name="neural", data=new_neural, electrodes=old_neural.electrodes, conversion=old_neural.conversion, timestamps=new_neural_timestamps))

    def _load_data_nwb(self):
        """
        Load data from NWB file

        """
        self.io = NWBHDF5IO(self.cfg.dataset_parameters.nwb_file, mode="r")
        self.dataset = self.io.read()

    def _load_data_zstruct(self):
        """
        Load data from zstruct format files.

        Uses either subject/date/run information or a direct data path to locate and load
        neural/emg recording data. Currently supports loading only a single run.

        Raises
        ------
        ValueError
            If required parameters are missing
        NotImplementedError
            If multiple runs are provided (current limitation)
        """
        # loading each run
        # FIXME: for now, failing if provided with more than one run
        # if len(self.cfg.runs) > 1:
        #     raise NotImplementedError(
        #         "Multi-run Datasets not supported (yet)"
        #     )

        # TODO: figure out a way of combining the runs into a single NWB file
        # for run in self.cfg.runs:
        #     # TODO: Check if run is already saved as nwb file and if it is just load it
        #     if self.verbose:
        #         print(f"\t+ Loading run {run}")
        # check is nwb file exists at location
        dpars = self.dataset_parameters
        nwb_exists = os.path.isfile(zstruct_loader.get_save_path(self.dataset_parameters))
        if nwb_exists:
            if dpars.overwrite:
                print("NWB file exists but overwriting")
                self.dataset = zstruct_loader.load_xpc_run(self.dataset_parameters)
            else:
                print("NWB file already exists, loading")
                self.io = NWBHDF5IO(zstruct_loader.get_save_path(self.dataset_parameters), mode="r")
                self.dataset = self.io.read()
                
        else:
            print("No existing NWB file, creating...")
            self.dataset = zstruct_loader.load_xpc_run(self.dataset_parameters)

    def save_data(self):
        """
        Save the dataset to an NWB file.

        Raises
        ------
        NotImplementedError
            If saving is not implemented for the dataset type
        ValueError
            If no path can be determined
        """
        if self.cfg.save_path == None:
            # if no custom location specified, save in default location according to dataset_type
            if self.cfg.dataset_type == "zstruct":
                path = zstruct_loader.get_save_path(self.dataset_parameters)
                print(path)
            elif self.cfg.dataset_type == "nwb":
                raise NotImplementedError("Saving NWB files is not implemented yet")
            else:
                raise NotImplementedError("dataset type not implemented")
        else:
            path = self.cfg.save_path
        print(f"Saving NWB file to {path}...")
        with NWBHDF5IO(path, mode="w") as io:
            io.write(self.dataset)

    def close(self):
        """
        Close the NWB file if it was opened.
        """
        if self.io:
            self.io.close()
            self.io = None

    def __del__(self):
        """
        Ensure the NWB file is closed when the object is destroyed.
        """
        self.close()