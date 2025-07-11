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
            autoload: book, specifies if the dataset should be automatically loaded when the dataset object is instantiated
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
            self.load_data_nwb()
        else:
            raise NotImplementedError(f"Unimplemented dataset type: {self.cfg.dataset_type}")
        
    def _initialize_nwb(self):
        """
        Initialize dataset for NWB type data.
        Currently a placeholder for future implementation.
        """
        pass

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

        self.dataset = zstruct_loader.load_xpc_run(self.cfg.dataset_parameters)

    def save_data(self, custom_path=None):
        """
        Save the dataset to an NWB file.

        Parameters
        ----------
        custom_path : str, optional
            Custom file path to save the NWB file. If None, a path will be generated
            based on the dataset type.

        Raises
        ------
        NotImplementedError
            If saving is not implemented for the dataset type
        ValueError
            If no path can be determined
        """
        path = custom_path if custom_path is not None else None
        if path is None and self.cfg.dataset_type == "zstruct":
            path = self._get_save_path_zstruct()
        elif path is None and self.cfg.dataset_type == "nwb":
            raise NotImplementedError("Saving NWB files is not implemented yet")
        elif path is None:
            raise ValueError("Path must be specified")

        print(f"Saving NWB file to {path}...")
        with NWBHDF5IO(path, mode="w") as io:
            io.write(self.dataset)

    def _get_save_path_zstruct(self):
        """
        Generate a file path for saving zstruct data as NWB.

        Returns
        -------
        str
            Full path to save the NWB file, including directory structure and filename
            based on subject, date, and run information
        """
        # Create a string representation of the runs
        runs_str = "_".join([f"Run-{run:03d}" for run in sorted(self.cfg.runs)])

        return os.path.join(
            self.server_dir,
            self.cfg.subject,
            self.cfg.date,
            f"{self.cfg.subject}_{self.cfg.date}_{runs_str}.nwb",
        )
