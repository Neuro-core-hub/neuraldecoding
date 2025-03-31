import os
import warnings
from sys import platform
import numpy as np
import re
import copy
import glob
from pynwb import NWBFile
from pynwb.file import Subject
from pynwb.device import Device
from pynwb.ecephys import (
    ElectrodeGroup,
    ElectricalSeries,
    SpikeEventSeries,
)
from pynwb.base import TimeSeries
from hdmf.backends.hdf5.h5_utils import H5DataIO
from datetime import datetime
from dateutil.tz import tzlocal
import heapq


# from ..utils import *
from ..utils.utils_general import get_creation_path_time, int_to_string, is_collection


def get_server_data_path(is_monkey=True, custom_server_folder=None):
    """
    Returns the server filepath based on what operating system is being used.

    Parameters
    ----------
    is_monkey : bool, default=True
        If True, the path will be set to the monkey data folder. If False, the path will be set to the human data folder.
    custom_server_folder : str, optional
        The custom server folder path. If provided, this will be used instead of the default paths.

    Returns
    -------
    str
        The complete path to the data directory.
    """
    # choose the standard path based on the OS
    if custom_server_folder is not None:
        server_data_path = f"{custom_server_folder}/Data"
    elif platform == "linux" or platform == "linux2":
        server_data_path = "/run/user/1000/gvfs/smb-share:server=cnpl-drmanhattan.engin.umich.edu,share=share/Data"
    elif platform == "darwin":
        server_data_path = "/Volumes/share/Data"
    elif platform == "win32":
        server_data_path = f"Z:/Data"

    if is_monkey:
        server_data_path = os.path.join(server_data_path, "Monkeys")
    else:
        server_data_path = os.path.join(server_data_path, "Humans", "RPNI")

    return server_data_path


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


# TODO: optimize this function - it's very slow right now


def read_xpc_data(contents, direc, xpc_dict, nwb_dict, verbose=False):
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
    # supported data types and their byte sizes #START REMOVE
    print("Running optimized code")
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
        [{"SpikeTimes": []} for _ in range(xpc_dict["num_channels"])]
        if spikeformat
        else None
    )

    base_dict = {
        **{name: None for name in fnames[::2]},
        "TrialNumber": None,
        **(
            {nwb_dict["signal_name"]: [], xpc_dict["spikes_field"]: channel_structure}
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
    for i in range(trials[-1]):
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

                dict_data[i][nwb_dict["signal_name"]] = neural_data

            except re.error:  # data is an empty cell
                dict_data[i][nwb_dict["signal_name"]] = []

    # Format Specific Fields
    ############################################################################

    # Change neural data field into spike times per channel (optimized for speed)
    if spikeformat:
        for i in range(trials[-1]):
            if i in dropped_list:  # skip dropped trials
                continue

            # Get references to avoid repeated dictionary lookups
            neural_data = dict_data[i][nwb_dict["signal_name"]]
            exp_time = dict_data[i][xpc_dict["experiment_time_field"]]
            neural_data_len = len(neural_data)

            # Use numpy for faster operations
            spikenums = np.zeros(
                (xpc_dict["num_channels"], neural_data_len), dtype=np.int32
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
            channel_data = dict_data[i][xpc_dict["spikes_field"]]
            for c in range(xpc_dict["num_channels"]):
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
            dict_data[i].pop(nwb_dict["signal_name"], None)

    # this next one removes the skipped trials
    # must use a generator because of indexing issues if you don't
    if (
        trials[-1] != ntrials
    ):  # only run if there is a trial that is dropped to save time
        dict_data = [trial for trial in dict_data if trial["TrialNumber"] is not None]

    # df_og=pd.DataFrame(dict_data)
    # #print(df_og.shape)  # Check if DataFrame has rows/columns
    # #print(df_og.head())  # Look at first few rows
    # try:
    #     df_og.to_csv('output_data_optimized.csv', index=False)
    #     print("File saved successfully")
    # except Exception as e:
    #     print(f"Error saving file: {e}")
    # #df_og.to_excel('original_data.xlsx',index=False)

    return dict_data


def initialize_nwb_columns(nwb_file, data_dict, xpc_dict):
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
        data_dict[-1][xpc_dict["experiment_time_field"]]
    )  # num of times for last trial - used to understand if a variable is a time-series
    num_total_times = sum(
        len(data_dict[trl_idx][xpc_dict["experiment_time_field"]])
        for trl_idx in range(num_trials)
    )

    for key in data_dict[0].keys():
        if key == "ExperimentTime":
            continue  # ExperimentTime are handled as special cases
        elif (
            key == xpc_dict["behavior_field"]
            or key == xpc_dict["signal_field"]
            or key == xpc_dict["spikes_field"]
        ):
            continue  # Behavior data and SBP are handled as special case (stored in the respective modules)
        elif (
            not is_collection(data_dict[0][key]) or len(data_dict[-1][key]) != num_times
        ):
            nwb_file.add_trial_column(name=key, description=key)
        else:
            time_series_dict[key] = np.empty(
                (num_total_times, data_dict[0][key].shape[1]),
                dtype=data_dict[0][key].dtype,
            )

    return time_series_dict


def initialize_nwb_file(
    data_path,
    run,
    creation_time,
    experimenter,
    notes_content,
    institution,
    lab,
    subject,
    device,
    electrode_group,
    xpc_dict,
):
    """
    Initialize an NWB file with metadata and electrode information.

    Parameters
    ----------
    data_path : str
        Path to the data directory.
    run : int
        Run number.
    creation_time : float
        Timestamp of when the data was created.
    experimenter : list
        List of experimenter names.
    notes_content : str
        Content of the notes file.
    institution : str
        Institution name.
    lab : str
        Lab name.
    subject : pynwb.file.Subject
        Subject information.
    device : pynwb.device.Device
        Recording device information.
    electrode_group : pynwb.ecephys.ElectrodeGroup
        Electrode group information.
    xpc_dict : dict
        Dictionary containing configuration parameters.

    Returns
    -------
    pynwb.NWBFile
        Initialized NWB file object.
    """

    # Get dataset details

    nwb_file = NWBFile(
        session_description=f"{data_path} | Run-{run:03d}",
        identifier=f"{data_path}/Run-{run:03d}",
        session_start_time=datetime.fromtimestamp(creation_time, tz=tzlocal()),
        file_create_date=datetime.now(tzlocal()),
        experimenter=experimenter,
        institution=institution,
        notes=notes_content,
        lab=lab,
        subject=subject,
        devices=[device],
        electrode_groups=[electrode_group],
    )

    for _ in range(xpc_dict["num_channels"]):
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
    xpc_dict,
    nwb_dict,
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
    num_behavior_vars = data_dict[0][xpc_dict["behavior_field"]].shape[1]

    total_times = sum(
        len(data_dict[trl_idx][xpc_dict["experiment_time_field"]])
        for trl_idx in range(num_trials)
    )

    # preallocate time-series data arrays
    times = np.empty(
        (total_times,), dtype=data_dict[0][xpc_dict["experiment_time_field"]].dtype
    )
    neural_data = np.empty(
        (total_times, xpc_dict["num_channels"]),
        dtype=data_dict[0][xpc_dict["signal_field"]].dtype,
    )
    behavior_data = np.empty(
        (total_times, num_behavior_vars),
        dtype=data_dict[0][xpc_dict["behavior_field"]].dtype,
    )

    # looping through the trials and populating the times-series data
    start_idx = 0

    for trl_idx in range(num_trials):
        times_trial_data = np.squeeze(
            data_dict[trl_idx][xpc_dict["experiment_time_field"]]
        )

        end_idx = start_idx + len(times_trial_data)

        times[start_idx:end_idx] = times_trial_data
        neural_data[start_idx:end_idx] = data_dict[trl_idx][xpc_dict["signal_field"]][
            :, : xpc_dict["num_channels"]
        ]
        behavior_data[start_idx:end_idx] = data_dict[trl_idx][
            xpc_dict["behavior_field"]
        ]

        # add all the other time-series data
        for key in time_series_dict.keys():
            time_series_dict[key][start_idx:end_idx] = data_dict[trl_idx][key]

        start_idx = end_idx

    # Set times to milliseconds
    times /= 1000

    # adding behavior data
    behavior_ts = TimeSeries(
        name=nwb_dict["behavior_name"],
        data=H5DataIO(
            behavior_data,
            compression=nwb_dict["compression"]["type"],
            compression_opts=nwb_dict["compression"]["options"],
        ),
        unit=nwb_dict["behavior_unit"],  # From 0 to 1
        timestamps=times,
        description=nwb_dict["behavior_description"],
        comments=nwb_dict["behavior_comments"],
    )

    # Add finger position as acquisition
    nwb_file.add_acquisition(behavior_ts)
    # nwb_modules["behavior"].add_data_interface(fingers_position_ts)

    # adding neural data
    neural_data_ts = ElectricalSeries(
        name=nwb_dict["signal_name"],
        data=H5DataIO(
            neural_data,
            compression=nwb_dict["compression"]["type"],
            compression_opts=nwb_dict["compression"]["options"],
        ),
        timestamps=times,
        description=nwb_dict["signal_description"],
        conversion=nwb_dict["signal_conversion"],
        channel_conversion=[1.0] * xpc_dict["num_channels"],
        filtering=nwb_dict["signal_filtering"],
        comments=nwb_dict["signal_comments"],
        electrodes=nwb_file.create_electrode_table_region(
            region=list(range(xpc_dict["num_channels"])),
            description=nwb_dict["electrode_group"]["description"],
        ),
    )

    # Add neural data as acquisition
    nwb_file.add_acquisition(neural_data_ts)
    # nwb_modules["neural_data"].add_data_interface(neural_data_ts)

    # adding all the other time-series data as acquisition data
    for key in time_series_dict.keys():
        nwb_file.add_acquisition(
            TimeSeries(
                name=key,
                unit="",
                data=H5DataIO(
                    time_series_dict[key],
                    compression=nwb_dict["compression"]["type"],
                    compression_opts=nwb_dict["compression"]["options"],
                ),
                timestamps=times,
                description=f"{key}",
                conversion=1.0,
            )
        )

    # Add spike series
    if xpc_dict["spikes_exist"]:
        spike_times = [[] for _ in range(xpc_dict["num_channels"])]
        for trial in data_dict:
            for ch_idx in range(xpc_dict["num_channels"]):
                spike_times[ch_idx].extend(
                    [trial[xpc_dict["spikes_field"]][ch_idx]["SpikeTimes"]]
                    if type(trial[xpc_dict["spikes_field"]][ch_idx]["SpikeTimes"])
                    is int
                    else trial[xpc_dict["spikes_field"]][ch_idx]["SpikeTimes"]
                )
            # Pop spikes field from data_dict
            trial.pop(xpc_dict["spikes_field"], None)
        all_spike_times = list(heapq.merge(*spike_times))
        spikes = np.zeros((len(all_spike_times), xpc_dict["num_channels"]))
        for ch_idx in range(xpc_dict["num_channels"]):
            # Find indices for each spike time
            spikes[
                np.searchsorted(all_spike_times, np.array(spike_times[ch_idx])), ch_idx
            ] = 1
        # Add spike series
        electrode_table = nwb_file.create_electrode_table_region(
            region=list(range(xpc_dict["num_channels"])),
            description=nwb_dict["electrode_group"]["description"],
        )
        spike_series = SpikeEventSeries(
            name=nwb_dict["spikes_name"],
            data=H5DataIO(
                spikes,
                compression=nwb_dict["compression"]["type"],
                compression_opts=nwb_dict["compression"]["options"],
            ),
            electrodes=electrode_table,
            timestamps=np.array(all_spike_times, dtype=float),
        )
        nwb_file.add_acquisition(spike_series)

    # looping through the trials for adding the trials and all the other non-timeseries data to the NWB file
    # base structure of each trial
    nwb_trial_dict = dict()

    # add the trials to the NWB file
    for trl_idx in range(num_trials):
        # trial times
        nwb_trial_dict["start_time"] = data_dict[trl_idx][
            xpc_dict["experiment_time_field"]
        ][0][0]
        nwb_trial_dict["stop_time"] = data_dict[trl_idx][
            xpc_dict["experiment_time_field"]
        ][-1][0]

        # looping through the data frame keys for dynamically adding them
        for key in data_dict[trl_idx].keys():
            if key == xpc_dict["experiment_time_field"]:
                pass  # ExperimentTime is handled as a special case
            elif (
                key == xpc_dict["behavior_field"]
                or key == xpc_dict["signal_field"]
                or key == xpc_dict["spikes_field"]
            ):
                pass
            elif key in time_series_dict.keys():
                pass
            else:  # non-time series data
                if key == "TrialNumber":
                    data_dict[trl_idx][
                        key
                    ] -= (
                        1  # subtract 1 since indexes in the matlab version start from 1
                    )

                nwb_trial_dict[key] = data_dict[trl_idx][key]

        nwb_file.add_trial(**nwb_trial_dict)


def get_nwb_from_run(
    subject,
    date,
    data_path,
    run,
    xpc_dict,
    nwb_dict,
    is_monkey,
):
    """
    Create and populate an NWB file from experimental run data.

    Parameters
    ----------
    subject : str or None
        Subject name. If None, it will be extracted from the data path.
    date : str
        Date of the recording.
    data_path : str or None
        Path to the data directory. If None, it will be constructed from server path.
    run : str or int
        Run identifier.
    xpc_dict : dict
        Dictionary containing configuration parameters for parsing the data.
    nwb_dict : dict
        Dictionary containing NWB-specific configuration parameters.
    is_monkey : bool
        If True, the path will be set to the monkey data folder. If False, the path will be set to the human data folder.

    Returns
    -------
    pynwb.NWBFile
        The populated NWB file object.
    """

    if data_path is None:
        data_path = os.path.join(
            get_server_data_path(is_monkey),
            subject,
            date,
        )

    # normalizing the path to be OS independent
    data_path = os.path.normpath(data_path)
    # Get dataset details
    subject, creation_time, experimenter, notes_content = get_dataset_details(
        data_path, subject, date
    )
    # Get the institution and lab
    institution = nwb_dict["institution"]
    lab = nwb_dict["lab"]
    # Get the device and electrode group
    device = Device(**dict(nwb_dict["device"]))
    electrode_group = ElectrodeGroup(device=device, **dict(nwb_dict["electrode_group"]))

    # Initialize the NWB file
    nwb_file = initialize_nwb_file(
        data_path,
        run,
        creation_time,
        experimenter,
        notes_content,
        institution,
        lab,
        subject,
        device,
        electrode_group,
        xpc_dict,
    )

    # Load the data from the server
    load_data_server(
        nwb_file,
        data_path,
        run,
        xpc_dict,
        nwb_dict,
    )

    return nwb_file


def load_data_server(
    nwb_file,
    data_path,
    run,
    xpc_dict,
    nwb_dict,
):
    """
    Load the data from the server and populate the NWB file with experimental data.

    Parameters
    ----------
    nwb_file : pynwb.NWBFile
        The NWB file object to populate with data.
    data_path : str
        Path to the data directory.
    run : int
        Run number to load.
    xpc_dict : dict
        Dictionary containing configuration parameters for parsing the data.
    nwb_dict : dict
        Dictionary containing NWB-specific configuration parameters.

    Returns
    -------
    list of dict
        List of dictionaries containing the processed data for each trial.
    """
    run_path = os.path.join(data_path, f"Run-{run:03d}")

    # loading the zScript file content
    contents = load_z_script(run_path)

    data_dict = read_xpc_data(contents, run_path, xpc_dict, nwb_dict)

    # initializing the nwb modules and columns
    time_series_dict = initialize_nwb_columns(nwb_file, data_dict, xpc_dict)

    # populate the nwb file with the run data
    add_run_data(
        nwb_file,
        data_dict,
        time_series_dict,
        xpc_dict,
        nwb_dict,
    )

    return data_dict
