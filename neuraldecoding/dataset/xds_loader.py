from pynwb import NWBHDF5IO, NWBFile
try:
    from xds_python import xds
    import fnmatch, os, sys
    HAS_XDS = True
except ImportError:
    HAS_XDS = False
    xds = None

def load_xds_to_nwb(dataset_parameters):
    if not HAS_XDS:
            raise ImportError("xds_python is not installed. Please install it to use this feature. Please install it by downloading content from https://github.com/limblab/xds/tree/master/xds_python into xds_python folder in neuraldecoding/dataset")
    fpath = dataset_parameters.fpath
    fname = dataset_parameters.fname
    if not os.path.isfile(os.path.join(fpath, fname)):
        raise ValueError(f"xds file not found at {os.path.join(fpath, fname)}")
    xds_data = xds.lab_data(fpath, fname)
    xds_data.update_bin_data(dataset_parameters.bin_size)
    xds_data.smooth_binned_spikes(dataset_parameters.bin_size, 'gaussian', dataset_parameters.smooth_size)
    data_unit_names = xds_data.unit_names
    #-------- Extract smoothed spike counts in trials without temporal alignment --------#
    data_spike = xds_data.get_trials_data_spike_counts('R', 'start_time', 0.0, 'end_time', 0)
    data_EMG_names = xds_data.EMG_names
    #-------- Extract EMG envelops in trials without temporal alignment --------#
    data_EMG = xds_data.get_trials_data_EMG('R', 'start_time', 0.0, 'end_time', 0)

    