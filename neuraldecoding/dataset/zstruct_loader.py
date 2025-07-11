import os
import warnings
from sys import platform
import numpy as np
import re
import copy
import pdb
from tqdm import tqdm
import glob
from pynwb import NWBFile
from pynwb.file import Subject
from pynwb.device import Device
from pynwb.ecephys import (
    ElectrodeGroup,
    ElectricalSeries,
    EventDetection,
)
from pynwb.base import TimeSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO
from datetime import datetime
from dateutil.tz import tzlocal
import heapq
from omegaconf import OmegaConf

# from ..utils import *
from ..utils.utils_general import get_creation_path_time, int_to_string, is_collection

# Registry of known types
SERIES_CLASSES = {
    "TimeSeries": TimeSeries,
    "ElectricalSeries": ElectricalSeries
}

def get_server_notes_details(data_path):
    """
    Extract experimenter information and notes content from the notes file in the data directory.

    Parameters
    ----------
    data_path : str
        Path to the data directory containing the notes file.

    Returns
    -------
    tuple
        A tuple containing (experimenter, notes_content) where experimenter is a list of names
        and notes_content is the full text of the notes file.
    """
    experiment_regex = "(?i)Experiment(er)?(s)?(:)?( )*"
    personnel_regex = "(?i)Personnel(:)?( )*"
    notes_regex = "(?i)Notes.*\.txt$"

    run_path_contents = os.listdir(data_path)

    for notes_file in run_path_contents:
        notes_file_path = os.path.join(data_path, notes_file)

        if re.search(notes_regex, notes_file) and os.path.isfile(notes_file_path):
            f = open(notes_file_path, "r")
            notes_content = f.read()
            f.close()
            line_count = -1
            personnel_found_line = -1

            for line in notes_content.splitlines():
                experiment_found = re.search(experiment_regex, line)
                personnel_found = re.search(personnel_regex, line)

                if personnel_found:
                    experimenter = (
                        line[personnel_found.end() :].replace(", ", ",").split(",")
                    )
                    personnel_found_line = line_count
                elif experiment_found and personnel_found_line == -1:
                    experimenter = (
                        line[experiment_found.end() :].replace(", ", ",").split(",")
                    )

                line_count += 1

    return experimenter, notes_content

def load_z_script(direc):
    """
    Load and read the zScript.txt file from the specified directory.

    Parameters
    ----------
    direc : str
        Directory path containing the zScript.txt file.

    Returns
    -------
    str
        Contents of the zScript.txt file.

    Raises
    ------
    Exception
        If the zScript.txt file is not found in the specified directory.
    """
    try:
        # load in and read z translator
        f = open(r"{}".format(os.path.join(direc, "zScript.txt")), "r")
        if f.mode == "r":
            contents = f.read()
            f.close()

            return contents
    except:
        raise Exception(
            "zScript.txt file not found. Make sure you're passing the right folder path."
        )

def read_xpc_data(direc, exp_cfg, verbose=False):
    """
    Parse the zScript contents and binary data files to create a structured dictionary of experimental data.

    Parameters
    ----------
    contents : str
        Contents of the zScript.txt file.
    direc : str
        Directory path containing the binary data files.
    xpc_dict : dict
        Dictionary containing configuration parameters for parsing the data.
    nwb_dict : dict
        Dictionary containing configuration parameters for the NWB file.
    verbose : bool, default=False
        If True, print additional information during processing.

    Returns
    -------
    list
        List of dictionaries, where each dictionary contains the data for one trial.
    """
    
    #load zscript.txt
    contents = load_z_script(direc)

    # supported data types and their byte sizes #START REMOVE
    cls = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 2,
        "uint32": 4,
        "int32": 4,
        "single": 4,
        "double": 8,
    }
    # data types and their python equivalent
    data_con = {
        "uint8": np.ubyte,
        "int8": np.byte,
        "uint16": np.ushort,
        "int16": np.short,
        "uint32": np.uintc,
        "int32": np.intc,
        "single": np.single,
        "double": np.double,
    }

    # Split Z string into its P/M/D/N substrings:
    zstr = re.split(":.:", contents)  # 1st entry is blank

    # Split each substring into its fieldname/datatype/numels substrings:
    for i in range(len(zstr)):
        zstr[i] = zstr[i].split("-")  # last entry in each cell is blank

    # Extract names, types, and sizes using list comprehension, starting from index 1
    names = [
        [zstr[i][j] for j in range(0, len(zstr[i]) - 1, 3)] for i in range(1, len(zstr))
    ]
    types = [
        [zstr[i][j] for j in range(1, len(zstr[i]) - 1, 3)] for i in range(1, len(zstr))
    ]
    sizes = [
        [zstr[i][j] for j in range(2, len(zstr[i]) - 1, 3)] for i in range(1, len(zstr))
    ]

    #######-----------------
    # Collect names, types, and sizes into list of list
    names1 = []
    types1 = []
    sizes1 = []
    for i in range(1, len(zstr)):
        names1.append([])
        types1.append([])
        sizes1.append([])
        for j in range(0, len(zstr[i]) - 1, 3):
            names1[i - 1].append(zstr[i][j])
        for j in range(1, len(zstr[i]) - 1, 3):
            types1[i - 1].append(zstr[i][j])
        for j in range(2, len(zstr[i]) - 1, 3):
            sizes1[i - 1].append(zstr[i][j])

    # Set flag(s) for specific field formatting:
    spikeformat = any("SpikeChans" in field_list for field_list in names)

    # Recover number of fields in each file type
    fnum = [len(field_list) for field_list in names]

    fnames = [None] * 2 * sum(fnum)
    # use ord() to change hexidecimal notation to correct int values
    bsizes = [[ord(size) for size in sublist] for sublist in sizes]

    # Calculate byte sizes for each feature and collect field names:
    m = 0
    for i, field_list in enumerate(names):
        for j, field_name in enumerate(field_list):
            fnames[m] = field_name
            m += 2
            try:
                bsizes[i][j] *= cls[types[i][j]]
            except KeyError as e:
                raise ValueError(
                    f"Unknown data type '{types[i][j]}' in cls dictionary."
                ) from e

    # Calculate bytes per timestep for each file:
    bytes = [int(np.sum(file_bytes) + 2) for file_bytes in bsizes]

    # Get number of trials in this run
    trial_list = glob.glob(os.path.join(direc, "tParams*"))
    ntrials = len(trial_list)

    # Extract and sort trial numbers more efficiently
    trials = sorted(
        [int(re.findall(r"\d+", trial_path)[-1]) for trial_path in trial_list]
    )

    # Check for dropped trials
    if trials[-1] != ntrials:
        warnings.warn("There is at least 1 dropped trial")

    # Initialize the dictionary with correct field names
    channel_structure = (
        [{"SpikeTimes": []} for _ in range(exp_cfg.num_channels)]
        if spikeformat
        else None
    )

    base_dict = {
        **{name: None for name in fnames[::2]},
        "TrialNumber": None,
        **(
            {'NeuralData': [], exp_cfg.units.field: channel_structure}
            if spikeformat
            else {}
        ),
    }

    # Create list of dictionaries for all trials
    dict_data = [copy.deepcopy(base_dict) for _ in range(trials[-1])]

    ############################## Parse Data Strings Into Dictionary: ######################################
    data = [[], [], [], []]  # initilize data
    dropped_list = []
    file_templates = [
        "tParams{}.bin",
        "mBehavior{}.bin",
        "dBehavior{}.bin",
        "neural{}.bin",
    ]
    for i in tqdm(range(trials[-1])):
        trial_num = i + 1
        dict_data[i]["TrialNumber"] = trial_num
        try:
            # add trial number to dict
            dict_data[i]["TrialNumber"] = i + 1

            # read in data files
            for file_idx, template in enumerate(file_templates):
                filepath = os.path.join(direc, template.format(trial_num))
                data[file_idx] = np.fromfile(filepath, dtype="uint8")

        except:
            dict_data[i][
                "TrialNumber"
            ] = None  # this will set up the removal of empty dictionaries
            dropped_list.append(i)  # sets up the spike formatting
            if verbose:
                print("Trial Number {} was dropped".format(i + 1))
            continue  # this skips to the next trial

        # Iterate through file types 1-3 and add data to Z:
        for j in range(4 - spikeformat):

            # Calculate # of timesteps in this file:
            nstep = len(data[j]) // bytes[j]

            # Calculate the byte offsets for each feature in the timestep:
            offs = np.concatenate(([3], 3 + np.cumsum(bsizes[j])))

            # Iterate through each field:
            for k in range(fnum[j]):
                # Create a byte mask for the uint8 data:
                # ----------xxxxxxxxxxxxxxxxxxx
                # mine
                mask_length = offs[k + 1] - offs[k]
                bmask = np.zeros(bytes[j], dtype=np.uint8)
                bmask[offs[k] - 1 : offs[k] + mask_length - 1] = 1
                bmask = np.tile(bmask, nstep)
                # -------------xxxxxxxxxxxxxxxxxxxxxx

                # ---------------xxxxxxxxxxxxxxxxx-------------------
                # Another try
                # Extract and reshape data in one step
                dat = data[j][bmask == 1].view(data_con[types[j][k]])

                # Directly assign reshaped data to dictionary (avoid creating temporary variables)
                dict_data[i][names[j][k]] = np.reshape(dat, (nstep, -1))

                # Simplify the data structure checking logic
                if len(dict_data[i][names[j][k]]) == 1:
                    # Avoid the second array access when checking length
                    first_element = dict_data[i][names[j][k]][0]
                    if len(first_element) == 1:
                        dict_data[i][names[j][k]] = first_element[0]  # Extract scalar
                    else:
                        dict_data[i][
                            names[j][k]
                        ] = first_element  # Extract single array
                # --------------xxxxxxx-----------------

        # Extract Neural data packets (optimized for speed)
        if spikeformat:
            new_string = int_to_string(data[3])  # convert to one continuous string

            # Pre-compute the search pattern only once
            search_char = chr((i + 1) % 256)
            if search_char in ".^$*+?{}[]|()\\":
                pattern = "\\" + int_to_string(
                    np.asarray([np.ushort(i + 1)]).view(np.ubyte)
                )
            else:
                pattern = int_to_string(np.asarray([np.ushort(i + 1)]).view(np.ubyte))

            # Single regex compilation and search
            pattern += "[^ÿ]*ÿ"
            try:
                ndata = re.findall(pattern, new_string)

                # Pre-allocate result array for better memory efficiency
                neural_data = []
                neural_data.extend([] for _ in range(len(ndata)))

                # Process all matches at once
                for m, match in enumerate(ndata):
                    content = match[2:-1]  # Extract content once
                    content_len = len(content)

                    if content_len == 0:
                        continue  # Keep empty list
                    elif content_len == 1:
                        neural_data[m] = [ord(content)]
                    else:
                        # Use a list comprehension instead of iterative appends
                        neural_data[m] = [ord(content[n]) for n in range(content_len)]

                dict_data[i]['NeuralData'] = neural_data

            except re.error:  # data is an empty cell
                dict_data[i]['NeuralData'] = []
    # Format Specific Fields
    ############################################################################

    # Change neural data field into spike times per channel (optimized for speed)
    if spikeformat:
        for i in range(trials[-1]):
            if i in dropped_list:  # skip dropped trials
                continue

            # Get references to avoid repeated dictionary lookups
            neural_data = dict_data[i]['NeuralData']
            exp_time = dict_data[i][exp_cfg.reference_time]
            neural_data_len = len(neural_data)

            # Use numpy for faster operations
            spikenums = np.zeros(
                (exp_cfg.num_channels, neural_data_len), dtype=np.int32
            )

            # Vectorize inner loop where possible
            for t in range(neural_data_len):
                data_point = neural_data[t]
                # Skip empty or zero-only arrays
                if not data_point or all(x == 0 for x in data_point):
                    continue

                for j in range(len(data_point)):
                    chan_idx = data_point[j] - 1  # -1 for zero-indexing
                    if chan_idx >= 0:  # Only increment for non-zero values
                        spikenums[chan_idx, t] += 1

            # Process each channel once
            channel_data = dict_data[i][exp_cfg.units.field]
            for c in range(exp_cfg.num_channels):
                # Only process channels with spikes
                spike_mask = spikenums[c, :] == 1
                if np.any(spike_mask):
                    # Get spike times directly with the mask
                    times = exp_time[spike_mask]

                    # Simplify the counting logic
                    spike_count = np.sum(spike_mask)

                    # Directly assign the spike times (transposed)
                    channel_data[c]["SpikeTimes"] = times.T

                    # Format optimization - handle scalar vs array cases
                    spike_times = channel_data[c]["SpikeTimes"]
                    if len(spike_times) == 1:
                        if len(spike_times[0]) != 1:
                            channel_data[c]["SpikeTimes"] = spike_times[0].astype(
                                np.int32
                            )
                        else:
                            channel_data[c]["SpikeTimes"] = int(
                                spike_times[0][0]
                            )  # Direct int conversion

            # Remove unneeded fields (moved outside the channel loop)
            dict_data[i].pop("SpikeChans", None)
            dict_data[i].pop("NeuralData", None)
            if not exp_cfg.units.has_spikes:
                dict_data[i].pop(exp_cfg.units.field)
    # this next one removes the skipped trials
    # must use a generator because of indexing issues if you don't
    if (
        trials[-1] != ntrials
    ):  # only run if there is a trial that is dropped to save time
        dict_data = [trial for trial in dict_data if trial["TrialNumber"] is not None]

    return dict_data

def initialize_nwb_columns(nwb_file, data_dict, exp_cfg):
    """
    Initialize the columns of the NWB file based on the dictionary keys and identify time series data.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
        The NWB file object to add columns to.
    data_dict : list of dict
        List of dictionaries containing the data for each trial.
    xpc_dict : dict
        Dictionary containing configuration parameters for parsing the data.

    Returns
    -------
    dict
        Dictionary of pre-allocated arrays for time series data.
    """

    # Dynamically create trial column and create the time_series dictionary
    time_series_dict = dict()

    num_trials = len(data_dict)
    num_times = len(
        data_dict[-1][exp_cfg.reference_time]
    )  # num of times for last trial - used to understand if a variable is a time-series
    num_total_times = sum(
        len(data_dict[trl_idx][exp_cfg.reference_time])
        for trl_idx in range(num_trials)
    )

    for key in data_dict[0].keys():
        if exp_cfg.units.has_spikes and key == exp_cfg.units.field:
            continue  # Spikes handled specially
        elif (not is_collection(data_dict[0][key]) or len(data_dict[-1][key]) != num_times):
            nwb_file.add_trial_column(name=key, description=key)
        else:
            time_series_dict[key] = np.concatenate([data_dict[i][key] for i in np.arange(len(data_dict))], axis=0, dtype=data_dict[0][key].dtype)
    
    return time_series_dict

def initialize_nwb_file(data_path, subject, date, run, exp_cfg):
    """
    Initialize an NWB file with metadata and electrode information.

    Parameters
    ----------
    data_path : str
        Path to the data directory.
    subject : str 
        Name of the subject
    date : str
        YYYY-MM-DD
    run: int
        run number
    num_channels:
        the number of channels/electrodes in the recording
    nwb_parameters:
        contains the other parameters for the nwb file creation
    
    Returns
    -------
    pynwb.NWBFile
        Initialized NWB file object.
    """
    
    # Get dataset details
    subject, creation_time, experimenter, notes_content = get_dataset_details(data_path, subject, date)

    # Get the device and electrode group
    device = Device(**dict(exp_cfg.device))
    electrode_group = ElectrodeGroup(device=device, **dict(exp_cfg.electrode_group))

    nwb_file = NWBFile(
        session_description=f"{data_path} | Run-{run:03d}",
        identifier=f"{data_path}/Run-{run:03d}",
        session_start_time=datetime.fromtimestamp(creation_time, tz=tzlocal()),
        file_create_date=datetime.now(tzlocal()),
        experimenter=experimenter,
        institution=exp_cfg.institution,
        notes=notes_content,
        lab=exp_cfg.lab,
        subject=subject,
        devices=[device],
        electrode_groups=[electrode_group],
    )

    for _ in range(exp_cfg.num_channels):
        nwb_file.add_electrode(group=electrode_group, location=electrode_group.location)

    return nwb_file

def get_dataset_details(data_path, subject, date):
    """
    Extract subject information, creation time, experimenter, and notes from the data directory.

    Parameters
    ----------
    data_path : str
        Path to the data directory.
    subject : str or None
        Subject name. If None, it will be extracted from the data path.
    date : str
        Date of the recording.

    Returns
    -------
    tuple
        A tuple containing (subject, creation_time, experimenter, notes_content) where:
        - subject is a pynwb.file.Subject object
        - creation_time is a timestamp
        - experimenter is a list of names
        - notes_content is the full text of the notes file

    Raises
    ------
    ValueError
        If data_path is None.
    """
    if data_path is None:
        raise ValueError("data_path is required")

    # retrieving the subject name from the data_path if not directly provided
    if subject is None:
        subject = data_path.split(os.sep)[-2]

    # setting the subject object
    subject = Subject(description=subject, subject_id=subject)

    # setting the creation time
    creation_time = get_creation_path_time(data_path)

    # setting the experimenter and notes content
    experimenter, notes_content = get_server_notes_details(data_path)
    return subject, creation_time, experimenter, notes_content

def add_run_data(
    nwb_file,
    data_dict,
    time_series_dict,
    exp_cfg,
    verbose=False,
):
    """
    Add experimental data to the NWB file, including neural, behavioral, and spike data.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
        The NWB file object to add data to.
    data_dict : list of dict
        List of dictionaries containing the data for each trial.
    time_series_dict : dict
        Dictionary of pre-allocated arrays for time series data.
    xpc_dict : dict
        Dictionary containing configuration parameters for the data.
    nwb_dict : dict
        Dictionary containing NWB-specific configuration parameters.
    verbose : bool, default=False
        If True, print additional information during processing.
    """

    num_trials = len(data_dict)

    # deal with time first
    times = time_series_dict[exp_cfg.reference_time].reshape(-1)
    times /= 1000 # set in seconds

    # create modules as defined by yaml file
    modules = {}
    for key in exp_cfg.modules:
        modules[key] = nwb_file.create_processing_module(name=exp_cfg.modules[key].name, description=exp_cfg.modules[key].description)

    # go through the time series and assign accordingly
    def create_ts(data, ts_cfg):
        cls = SERIES_CLASSES.get(ts_cfg["nwb_type"])
        params = dict(ts_cfg["nwb_params"])
        if ts_cfg["nwb_type"] == 'ElectricalSeries':
            params["electrodes"] = nwb_file.create_electrode_table_region(
                region=list(range(exp_cfg.num_channels)),
                description = exp_cfg.electrode_group.description
            )
        data_ts = cls(
            data=H5DataIO(data,
                          compression=exp_cfg.compression.type,
                          compression_opts=exp_cfg.compression.options),
            timestamps = times,
            **params
            )
        return data_ts
        

    for key in time_series_dict.keys():
        if any(named_series == key for named_series in exp_cfg.timeseries.keys()):
            if exp_cfg.timeseries[key]["nwb_type"] == "MultiFeature":
                for i in np.arange(exp_cfg.timeseries[key]["num_features"]):
                    data = time_series_dict[key][:,i*exp_cfg.num_channels:i*exp_cfg.num_channels + exp_cfg.num_channels]
                    feature = exp_cfg.timeseries[key][f"feat{i}"]
                    data_ts = create_ts(data, feature)
                    modules[exp_cfg.timeseries[key]["module"]].add(data_ts)
            else:
                data_ts = create_ts(time_series_dict[key], exp_cfg.timeseries[key])
                modules[exp_cfg.timeseries[key]["module"]].add(data_ts)
        elif key == exp_cfg.reference_time:
            continue
        else:
            #otherwise just add to acquisition as a timeseries
            data_ts = TimeSeries(
                name = key,
                description = key,
                timestamps = times,
                unit = "AU",
                data = H5DataIO(time_series_dict[key],
                              compression=exp_cfg.compression.type,
                              compression_opts=exp_cfg.compression.options)
            )
            Warning(f"Timeseries {key} was not defined in config file, adding to acquisitions")
            nwb_file.add_acquisition(data_ts)
            # undefined acquisition throw warning
            
    # Add spike series
    # pdb.set_trace()
    if exp_cfg.units.has_spikes:
        nwb_file.add_unit_column(name="channel", description="channel_id")
        all_events = []
        for ch_idx in range(exp_cfg.num_channels):
            cross_trial_spike_times = []
            for trial in data_dict:
                if type(trial[exp_cfg.units.field][ch_idx]["SpikeTimes"]) is int:
                    trial_spike_times = [trial[exp_cfg.units.field][ch_idx]["SpikeTimes"]]
                    cross_trial_spike_times.append(trial_spike_times)
                elif len(trial[exp_cfg.units.field][ch_idx]["SpikeTimes"]) > 0:
                    trial_spike_times = trial[exp_cfg.units.field][ch_idx]["SpikeTimes"]
                    cross_trial_spike_times.append(trial_spike_times)

            # pdb.set_trace()
            spike_times = np.concat(cross_trial_spike_times)
            nwb_file.add_unit(spike_times=spike_times, channel=ch_idx)

    # looping through the trials for adding the trials and all the other non-timeseries data to the NWB file
    # base structure of each trial
    nwb_trial_dict = dict()

    # add the trials to the NWB file
    for trl_idx in range(num_trials):
        # trial times
        nwb_trial_dict["start_time"] = data_dict[trl_idx][exp_cfg.reference_time][0][0]
        nwb_trial_dict["stop_time"] = data_dict[trl_idx][exp_cfg.reference_time][-1][0]

        # looping through the data frame keys for dynamically adding them
        for key in data_dict[trl_idx].keys():
            if key == exp_cfg.reference_time or key in time_series_dict.keys() or (exp_cfg.units.has_spikes and key == exp_cfg.units.field):
                pass
            else:  # non-time series data
                if key == "TrialNumber":
                    data_dict[trl_idx][key] -= (1)  # subtract 1 since indexes in the matlab version start from 1
                nwb_trial_dict[key] = data_dict[trl_idx][key]

        nwb_file.add_trial(**nwb_trial_dict)


def load_xpc_run(cfg):
    """
    Create and populate an NWB file from experimental run data.

    Parameters
    ----------
    cfg : DictConfig, the 'dataset_parameters' section of a Dataset yaml file containing:
            experiment_type: str
                Since we do different experiments with different features, this points to the right config file for that experiment type
            server_dir: str
                the location of the CNPL server on this machine
            alt_filepath: str
                if the binary files are located somewhere other than the server and not organized in the same way (subject/date/run) then an alternate filepath can be included here.
            subject: str
                name of the subject
            date: str - YYYY-MM-DD
                date of the experiment
            runs: int
                the run you are currently loading

    Returns
    -------
    pynwb.NWBFile
        The populated NWB file object.
    """
    # load the config of the particular experiment type
    config_dir = os.path.join(os.path.dirname(__file__), 'loader_configs', 'zstruct')
    config_path = os.path.join(os.path.join(config_dir, f'{cfg.experiment_type}.yaml'))

    exp_cfg = OmegaConf.load(config_path)

    # TODO: loading multiple runs
    if cfg.alt_filepath is not None:
        data_path = os.path.join(cfg.server_dir, cfg.subject, cfg.date)
    else:
        data_path = os.path.join(cfg.alt_filepath)

    print(f"Loading Run {cfg.runs} from {cfg.subject_id} on {cfg.date}")
    # normalizing the path to be OS independent
    data_path = os.path.normpath(data_path)
    run_path = os.path.join(data_path, f"Run-{cfg.runs:03d}")

    # Initialize the NWB file
    nwb_file = initialize_nwb_file(data_path, cfg.subject, cfg.date, cfg.runs, exp_cfg)

    # Read in the binary files
    data_dict = read_xpc_data(run_path, exp_cfg)

    # initializing the nwb modules and columns
    time_series_dict = initialize_nwb_columns(nwb_file, data_dict, exp_cfg)

    # populate the nwb file with the run data
    add_run_data(
        nwb_file,
        data_dict,
        time_series_dict,
        exp_cfg
    )

    return nwb_file
