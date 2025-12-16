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
        self.dataset_ratio = 1.0 # will be inferred from neural data
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
        self.dataset = self.copy_nwb_contents(self.io.read())
        dataset_sizes = [len(self.dataset.acquisition["neural"].timestamps[:])]
        for idx, nwb_file in enumerate(self.cfg.dataset_parameters.nwb_files[1:]):
            idx = idx + 1  # since we skipped the first file
            if self.cfg.dataset_parameters.continuous_add_mode == "same_dofs":
                continuous_filters = self.cfg.dataset_parameters.get("continuous_filters", [])

            # Read the NWB file
            with NWBHDF5IO(nwb_file, mode="r") as io:
                nwbdata = io.read()
                # Assume there is neural data instantly at the start of each run
                timestamp_offset = nwbdata.acquisition["neural"].timestamps[0] - self.dataset.acquisition["neural"].timestamps[-1]
                
                # Adjust trial times if trials exist
                if hasattr(nwbdata, 'trials') and nwbdata.trials is not None and len(nwbdata.trials) > 0:
                    # Get the trial data from current file
                    trials_df = nwbdata.trials.to_dataframe()
                    
                    # Get the offset for trial_count based on existing trials
                    trial_count_offset = len(self.dataset.trials) if hasattr(self.dataset, 'trials') and self.dataset.trials is not None else 0
                    
                    # Adjust the time columns
                    if 'start_time' in trials_df.columns:
                        trials_df['start_time'] = trials_df['start_time'] - timestamp_offset
                    if 'stop_time' in trials_df.columns:
                        trials_df['stop_time'] = trials_df['stop_time'] - timestamp_offset
                    if 'cue_time' in trials_df.columns:
                        trials_df['cue_time'] = trials_df['cue_time'] - timestamp_offset
                    if 'trial_count' in trials_df.columns:
                        trials_df['trial_count'] = trials_df['trial_count'] + trial_count_offset
                    
                    # Add adjusted trials to the dataset
                    for _, trial in trials_df.iterrows():
                        self.dataset.add_trial(**trial.to_dict())

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

                    dataset_sizes.append(len(nwbdata.acquisition["neural"].timestamps[:]))
                    
                    # Get neural from current nwb file
                    additional_neural = nwbdata.acquisition["neural"].data[:]
                    new_neural = np.vstack([new_neural, additional_neural])
                    # Modify timestamps
                    new_neural_timestamps = np.concatenate([old_neural.timestamps[:], nwbdata.acquisition["neural"].timestamps[:] - timestamp_offset])
                    self.dataset.add_acquisition(ElectricalSeries(name="neural", data=new_neural, electrodes=old_neural.electrodes, conversion=old_neural.conversion, timestamps=new_neural_timestamps))
                if self.cfg.dataset_parameters.continuous_add_mode == "same_dofs":
                    old_continuous = self.dataset.acquisition.pop("continuous")
                    if idx == 1:
                        new_continuous = old_continuous.data[:, continuous_filters[0]]
                    else:
                        new_continuous = old_continuous.data[:]
                    
                    # Get neural from current nwb file
                    additional_continuous = nwbdata.acquisition["continuous"].data[:, continuous_filters[idx]]

                    new_continuous = np.vstack([new_continuous, additional_continuous])

                    # Modify timestamps
                    new_continuous_timestamps = np.concatenate([old_continuous.timestamps[:], nwbdata.acquisition["continuous"].timestamps[:] - timestamp_offset])
                    self.dataset.add_acquisition(TimeSeries(name="continuous", data=new_continuous, timestamps=new_continuous_timestamps, description=old_continuous.description, unit=old_continuous.unit))

        dataset_sizes = np.array(dataset_sizes)
        self.dataset_ratio = dataset_sizes / np.sum(dataset_sizes)
        print(f"Dataset ratio overridden: {self.dataset_ratio}")
        
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

    @staticmethod
    def copy_nwb_contents(source_nwb: NWBFile) -> NWBFile:
        """
        Copy all contents from a source NWB file to a new NWB file.
        
        Parameters
        ----------
        source_nwb : NWBFile
            The source NWB file to copy from
            
        Returns
        -------
        NWBFile
            A new NWB file with copied contents
        """
        # Create new NWB file with source metadata
        target_nwb = NWBFile(
            session_description=source_nwb.session_description,
            identifier=source_nwb.identifier,
            session_start_time=source_nwb.session_start_time,
        )
        
        # Copy acquisition data
        for name, timeseries in source_nwb.acquisition.items():
            target_nwb.add_acquisition(timeseries)
        
        # Copy processing modules
        for module_name, module in source_nwb.processing.items():
            target_nwb.create_processing_module(
                name=module_name,
                description=module.description
            )
            for container_name, container in module.data_interfaces.items():
                target_nwb.processing[module_name].add(container)
        
        # Copy trials
        if hasattr(source_nwb, 'trials') and source_nwb.trials is not None:
            # First, add all trial columns from source
            for col in source_nwb.trials.colnames:
                if col not in ['start_time', 'stop_time']:  # These are default columns
                    col_data = source_nwb.trials[col]
                    target_nwb.add_trial_column(
                        name=col,
                        description=col_data.description if hasattr(col_data, 'description') else ''
                    )
            
            # Now add the trial data
            trials_df = source_nwb.trials.to_dataframe()
            for _, trial in trials_df.iterrows():
                target_nwb.add_trial(**trial.to_dict())
        
        # Copy electrodes
        if hasattr(source_nwb, 'electrodes') and source_nwb.electrodes is not None:
            electrodes_df = source_nwb.electrodes.to_dataframe()
            for _, electrode in electrodes_df.iterrows():
                target_nwb.add_electrode(**electrode.to_dict())
        
        # Copy units
        if hasattr(source_nwb, 'units') and source_nwb.units is not None:
            units_df = source_nwb.units.to_dataframe()
            for _, unit in units_df.iterrows():
                target_nwb.add_unit(**unit.to_dict())
        
        return target_nwb