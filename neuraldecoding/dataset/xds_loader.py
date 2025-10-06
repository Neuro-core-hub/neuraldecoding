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
from pynwb.misc import AnnotationSeries
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
    #   "day0_EMG": list of arrays, each array is a trial. Each array is (number of time bins, number of EMG channels)
    if not HAS_XDS:
        raise ImportError("xds_python is not installed. Please install it to use this feature. Please install it by downloading content from https://github.com/limblab/xds/tree/master/xds_python into xds_python folder in neuraldecoding/dataset")
    day0_path = dataset_parameters.fpath
    day0_name = dataset_parameters.fname

    if not os.path.isfile(os.path.join(day0_path, day0_name)):
        raise ValueError(f"xds file not found at {os.path.join(day0_path, day0_name)}")
    day0_data = xds.lab_data(day0_path, day0_name)

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

    # Create NWB file
    nwb_file = NWBFile(
        identifier=f"xds_data_{day0_name}",
        session_description=f"XDS neural and EMG data from {day0_name}",
        lab="limblab",
        session_start_time=datetime.now(tz=tzlocal()),
    )
    
    # Create a device for neural recordings
    device = Device(name="neural_array", description="array")
    nwb_file.add_device(device)
    
    # Create electrode group
    electrode_group = ElectrodeGroup(
        name="electrodes", 
        description="electrodes",
        location="brain",
        device=device
    )
    nwb_file.add_electrode_group(electrode_group)
    
    # Add electrodes to the electrode table
    all_unit_names = np.sort(list(set(day0_unit_names)))
    for i, unit_name in enumerate(all_unit_names):
        nwb_file.add_electrode(
            x=0.0, y=0.0, z=0.0,  # placeholder coordinates
            imp=np.nan,
            location="motor cortex",
            filtering="none",
            group=electrode_group,
            id=i
            # Note: unit names will be stored in metadata module instead of electrode labels
        )
    
    # Concatenate all day0 spike trials for NWB storage
    day0_spike_concat = np.concatenate(day0_spike, axis=0)
    
    # Do the same for EMG data
    day0_EMG_concat = np.concatenate(day0_EMG, axis=0)
    
    # Calculate trial start and end times (bin indices) - should be same for both spike and EMG
    day0_trial_start_times = []
    day0_trial_end_times = []
    cumulative_bins = 0
    
    print(f"Day0 Trials: {len(day0_spike)} spike trials, {len(day0_EMG)} EMG trials")
    
    # Verify that spike and EMG have same number of trials and same lengths
    if len(day0_spike) != len(day0_EMG):
        print(f"WARNING: Mismatch in trial count - Spike: {len(day0_spike)}, EMG: {len(day0_EMG)}")
    
    for trial_idx in range(len(day0_spike)):
        spike_length = day0_spike[trial_idx].shape[0]
        emg_length = day0_EMG[trial_idx].shape[0]
        
        if spike_length != emg_length:
            print(f"WARNING: Trial {trial_idx} length mismatch - Spike: {spike_length}, EMG: {emg_length}")
        
        # Use spike trial length as reference (they should be the same)
        start_bin = cumulative_bins
        end_bin = cumulative_bins + spike_length - 1  # -1 because end is inclusive
        
        day0_trial_start_times.append(start_bin)
        day0_trial_end_times.append(end_bin)
        
        cumulative_bins += spike_length
        
    # Create electrical series for spike data
    electrode_table_region = nwb_file.create_electrode_table_region(
        region=list(range(len(all_unit_names))),
        description="electrodes for spike data"
    )
    
    spike_electrical_series = ElectricalSeries(
        name="spike",
        data=H5DataIO(day0_spike_concat, compression=True),
        electrodes=electrode_table_region,
        starting_time=0.0,
        rate=1.0/bin_size,  # sampling rate based on bin size
        description="Binned and smoothed spike counts for day 0"
    )
    nwb_file.add_acquisition(spike_electrical_series)
    
    # Create time series for EMG data with channel names
    emg_time_series = TimeSeries(
        name="emg",
        data=H5DataIO(day0_EMG_concat, compression=True),
        unit="normalized",
        starting_time=0.0,
        rate=1.0/bin_size,  # sampling rate based on bin size
        description=f"EMG envelope data for day 0. Channels: {', '.join(day0_EMG_names)}"
    )
    nwb_file.add_acquisition(emg_time_series)
    
    # Create a processing module for metadata
    metadata_module = nwb_file.create_processing_module(
        name='metadata', 
        description='Custom metadata including unit names and EMG channel names'
    )
    
    # Store unit names as annotation
    unit_names_annotation = AnnotationSeries(
        name="unit_names",
        data=day0_unit_names,
        timestamps=[0.0] * len(day0_unit_names),  # dummy timestamps
        description="Neural unit names corresponding to electrode indices"
    )
    metadata_module.add(unit_names_annotation)
    
    # Store EMG channel names as annotation  
    emg_names_annotation = AnnotationSeries(
        name="emg_names",
        data=day0_EMG_names,
        timestamps=[0.0] * len(day0_EMG_names),  # dummy timestamps
        description="EMG channel names corresponding to EMG data columns"
    )
    metadata_module.add(emg_names_annotation)
    
    # Store trial timing information
    trial_starts_annotation = AnnotationSeries(
        name="trial_start_bins",
        data=day0_trial_start_times,
        timestamps=[0.0] * len(day0_trial_start_times),
        description="Start bin indices for each trial in concatenated data"
    )
    metadata_module.add(trial_starts_annotation)

    trial_ends_annotation = AnnotationSeries(
        name="trial_end_bins",
        data=day0_trial_end_times,
        timestamps=[0.0] * len(day0_trial_end_times),
        description="End bin indices for each trial in concatenated data"
    )
    metadata_module.add(trial_ends_annotation)
    
    return nwb_file