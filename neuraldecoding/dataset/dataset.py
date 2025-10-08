import os
from pynwb import NWBHDF5IO, NWBFile
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
        else:
            raise NotImplementedError(f"Unimplemented dataset type: {self.cfg.dataset_type}")
    
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