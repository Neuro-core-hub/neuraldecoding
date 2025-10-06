from pynwb import NWBHDF5IO, NWBFile
from omegaconf import OmegaConf
import neuraldecoding.dataset.zstruct_loader  as zstruct_loader
from pynwb import NWBFile
from pynwb.file import Subject
from pynwb.device import Device
from pynwb.ecephys import (
    ElectrodeGroup,
    ElectricalSeries,
    EventDetection,
)
from pynwb.base import TimeSeries
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb.base import TimeSeries
import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO

import fnmatch, os, sys

try:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(file_dir, 'xds_python'))
    import xds
    HAS_XDS = True
except ImportError:
    HAS_XDS = False
    xds = None


def emg_preprocessing(trial_emg, EMG_names):
    """
    Function to clip and normalize each channel of EMG envelopes
    :param trial_emg: list of lists containing EMG envelopes for each successful trial
    :return: list of lists containing the pre-processed EMG envelopes for each successful trial
    """
    
    # First, keep EMG channels that are good across all recorded sessions
    EMG_names_good = ['EMG_ECRb', 'EMG_ECRl', 'EMG_ECU', 'EMG_EDCr', 'EMG_FCR', 'EMG_FCU', 'EMG_FDP'] 
    idx_emg = [EMG_names.index(x) for x in EMG_names_good]
    for trial, val_trial in enumerate(trial_emg):
        trial_emg[trial] = val_trial[:, idx_emg]
    
    trial_emg_np = np.concatenate(trial_emg)

    # EMG clipping
    outlier = np.mean(trial_emg_np, axis=0) + 6 * np.std(trial_emg_np, axis=0)
    for i, val in enumerate(trial_emg):
        for ii in range(len(outlier)):
            trial_emg[i][:, ii] = trial_emg[i][:, ii].clip(max=outlier[ii])

    # EMG normalization
    trial_emg_np_baseline = np.percentile(trial_emg_np, 2, axis=0)
    trial_emg_np_max = np.percentile(trial_emg_np - trial_emg_np_baseline, 90, axis=0)
    for i, val in enumerate(trial_emg):
        trial_emg[i] = (val - trial_emg_np_baseline) / trial_emg_np_max

    return trial_emg, EMG_names_good

def spike_preprocessing(unit_names1, unit_names2, spike1, spike2):
    """
    unit_names1: unit names in the first dataset
    unit_names2: unit names in the second dataset
    spike1: a list, the spike counts for each trial in the first dataset
    spike2: a list, the spike counts for each trial in the second dataset
    """
    all_unit_names = np.sort(list(set(unit_names1)|set(unit_names2)))
    N_unit = len(all_unit_names)
    
    idx = [list(all_unit_names).index(e) for e in unit_names1]
    spike1_ = [np.zeros((s.shape[0], N_unit)) for s in spike1]
    for k in range(len(spike1)):
        spike1_[k][:, idx] = spike1[k]
        
    idx = [list(all_unit_names).index(e) for e in unit_names2]
    spike2_ = [np.zeros((s.shape[0], N_unit)) for s in spike2]
    for k in range(len(spike2)):
        spike2_[k][:, idx] = spike2[k]
    return spike1_, spike2_

def load_xds(dataset_parameters):
    # outputs dictionary containing keys:
    #   "day0_spike": list of arrays, each array is a trial. Each array is (number of time bins, number of channels/electrodes)
    #   "dayk_spike": list of arrays, each array is a trial. Each array is (number of time bins, number of channels/electrodes)
    #   "day0_EMG": list of arrays, each array is a trial. Each array is (number of time bins, number of EMG channels)
    #   "dayk_EMG": list of arrays, each array is a trial. Each array is (number of time bins, number of EMG channels)
    if not HAS_XDS:
        raise ImportError("xds_python is not installed. Please install it to use this feature. Please install it by downloading content from https://github.com/limblab/xds/tree/master/xds_python into xds_python folder in neuraldecoding/dataset")
    day0_path = dataset_parameters.day0_path
    day0_name = dataset_parameters.day0_name
    dayk_path = dataset_parameters.dayk_path
    dayk_name = dataset_parameters.dayk_name

    if not os.path.isfile(os.path.join(day0_path, day0_name)):
        raise ValueError(f"xds file not found at {os.path.join(day0_path, day0_name)}")
    day0_data = xds.lab_data(day0_path, day0_name)
    if not os.path.isfile(os.path.join(dayk_path, dayk_name)):
        raise ValueError(f"xds file not found at {os.path.join(dayk_path, dayk_name)}")
    dayk_data = xds.lab_data(dayk_path, dayk_name)

    bin_size = dataset_parameters.bin_size
    smooth_size = dataset_parameters.smooth_size

    #============================================= Load day-0 data ==================================================================#
    day0_data.update_bin_data(bin_size)  # Bin the spikes with the specified bin_size
    day0_data.smooth_binned_spikes(bin_size, 'gaussian', smooth_size) # Smooth the binned spike counts
    day0_unit_names = day0_data.unit_names
    #-------- Extract smoothed spike counts in trials without temporal alignment --------#
    day0_spike = day0_data.get_trials_data_spike_counts('R', 'start_time', 0.0, 'end_time', 0)
    day0_EMG_names = day0_data.EMG_names
    #-------- Extract EMG envelops in trials without temporal alignment --------#
    day0_EMG = day0_data.get_trials_data_EMG('R', 'start_time', 0.0, 'end_time', 0)
    day0_EMG, day0_EMG_names = emg_preprocessing(day0_EMG, day0_EMG_names) # outlier removal and normalization

    #============================================= Load day-k data ==================================================================#
    dayk_data.update_bin_data(bin_size)  
    dayk_data.smooth_binned_spikes(bin_size, 'gaussian', smooth_size)
    dayk_unit_names = dayk_data.unit_names
    #-------- Extract smoothed spike counts in trials without temporal alignment --------#
    dayk_spike = dayk_data.get_trials_data_spike_counts('R', 'start_time', 0.0, 'end_time', 0)
    dayk_EMG_names = dayk_data.EMG_names
    #-------- Extract EMG envelops in trials without temporal alignment --------#
    dayk_EMG = dayk_data.get_trials_data_EMG('R', 'start_time', 0.0, 'end_time', 0)
    dayk_EMG, dayk_EMG_names = emg_preprocessing(dayk_EMG, dayk_EMG_names)

    #============================================= Pre-processing ==================================================================#
    day0_spike, dayk_spike = spike_preprocessing(day0_unit_names, dayk_unit_names, day0_spike, dayk_spike) # zero-padding empty channels

    nwb_file = NWBFile(
        identifier=f"xds_data_{day0_name}-{dayk_name}",
        session_description=f"xds data: {day0_name}",
        lab="limblab",
        session_start_time=datetime.now(tz=tzlocal()),
    )
    modules = {}
    modules['ecephys'] = nwb_file.create_processing_module(name='ecephys', description='Ecephys processed data')

    def create_ts(data, name):
        data_ts = ElectricalSeries(name=name, data=H5DataIO(data))
        return data_ts
    modules['ecephys'].add(create_ts(np.array(day0_spike[0]).T, "day0_spike"))
    
    # Write the NWB file to disk and display information about it
    with NWBHDF5IO('xds_data.nwb', 'w') as io:
        io.write(nwb_file)

    # Read and display the NWB file contents
    with NWBHDF5IO('xds_data.nwb', 'r') as io:
        read_nwb = io.read()
        print("NWB File Contents:")
        print(f"Session Description: {read_nwb.session_description}")
        print(f"Lab: {read_nwb.lab}")
        print(f"Processing Modules: {list(read_nwb.processing.keys())}")
        if 'ecephys' in read_nwb.processing:
            ecephys_module = read_nwb.processing['ecephys']
            print(f"Ecephys Module Contents: {list(ecephys_module.data_interfaces.keys())}")
    return {
            "day0_spike": day0_spike,
            "dayk_spike": dayk_spike,
            "day0_EMG": day0_EMG,
            "dayk_EMG": dayk_EMG,
            }