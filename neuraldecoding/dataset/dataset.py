import os
from pynwb import NWBHDF5IO, NWBFile
from datetime import datetime
from dateutil.tz import tzlocal

from .utils_zstruct import (
    get_server_data_path,
    get_nwb_from_run,
)

from omegaconf import DictConfig


class Dataset:
    def __init__(self, cfg: DictConfig, verbose=True):
        """
        Initialize a Dataset object for neural data processing.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object containing at least the following dataset parameters:
            - type: str - Dataset type ('zstruct', 'nwb')
            - is_monkey: bool - Whether data is from monkey (True) or human (False)
            - subject: str - Subject identifier
            - date: str - Recording date
            - data_path: str - Alternative path to data
            - runs: list - List of run numbers to load
            - xpc: dict - XPC file configuration
            - nwb: dict - NWB file configuration
        verbose : bool, default=True
            Whether to print status messages during processing
        """
        self.cfg: DictConfig = cfg
        self.verbose: bool = verbose
        # Initialize empty NWB file
        self.dataset: NWBFile = NWBFile(
            session_description="",
            identifier="",
            session_start_time=datetime.now(tzlocal()),
        )

        if self.cfg.type == "zstruct":
            self._initialize_zstruct()
        elif self.cfg.type == "nwb":
            self._initialize_nwb()
        else:
            raise NotImplementedError(f"Unimplemented dataset type: {self.cfg.type}")

    def _initialize_zstruct(self):
        """
        Initialize dataset for zstruct type data.
        Sets up the server directory path based on configuration.
        """
        # self.institution, self.lab = get_creator_details()
        # self.device, self.electrode_group = get_device_and_electrode_group(data_type)
        self.server_dir = get_server_data_path(is_monkey=self.cfg.is_monkey)

    def _initialize_nwb(self):
        """
        Initialize dataset for NWB type data.
        Currently a placeholder for future implementation.
        """
        pass

    def load_data(self):
        """
        Load data based on the dataset type specified in configuration.
        Delegates to the appropriate loading method based on dataset type.

        Raises
        ------
        NotImplementedError
            If the dataset type is not supported
        """
        if self.cfg.type == "zstruct":
            self._load_data_zstruct()
        elif self.cfg.type == "nwb":
            self.load_data_nwb()
        else:
            raise NotImplementedError(f"Unimplemented dataset type: {self.cfg.type}")

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
        if (
            self.cfg.subject is None or self.cfg.date is None
        ) and self.cfg.data_path is None:
            raise ValueError(
                "Either ('subject' and 'date') or 'data_path' must be specified"
            )

        if self.cfg.runs is None:
            raise ValueError("Parameter 'runs' must be specified")

        if self.verbose:
            print(
                f"Loading data for {self.cfg.subject} at {self.cfg.date}, runs {', '.join(map(str, self.cfg.runs))}"
            )

        # loading each run
        # FIXME: for now, failing if provided with more than one run
        if len(self.cfg.runs) > 1:
            raise NotImplementedError(
                "For now, failing if provided with more than one run"
            )

        # TODO: figure out a way of combining the runs into a single NWB file
        for run in self.cfg.runs:
            # TODO: Check if run is already saved as nwb file and if it is just load it
            if self.verbose:
                print(f"\t+ Loading run {run}")

            self.dataset = get_nwb_from_run(
                subject=self.cfg.subject,
                date=self.cfg.date,
                data_path=self.cfg.data_path,
                run=run,
                xpc_dict=self.cfg.xpc,
                nwb_dict=self.cfg.nwb,
                is_monkey=self.cfg.is_monkey,
            )

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
        if path is None and self.cfg.type == "zstruct":
            path = self._get_save_path_zstruct()
        elif path is None and self.cfg.type == "nwb":
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
