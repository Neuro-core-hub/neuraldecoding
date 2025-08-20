import numpy as np
from pynwb import NWBFile
from .utils_general import resolve_path

def success_rate(nwb_file: NWBFile, success_field: str = 'success', exclude_trials: list|np.ndarray|None = None) -> float:
    """
    Calculate the success rate of the task.
    
    Parameters:
    -----------
    nwb_file : NWBFile
        The NWB file containing trial data
    success_field : str, optional
        The field name for success data (default: 'success')
    exclude_trials : list, optional
        List of trial indices to exclude from calculation
    
    Returns:
    --------
    float
        The success rate as a proportion (0-1)
    """
    success_data = nwb_file.trials[success_field][:]
    
    if exclude_trials is not None:
        # Create a mask to exclude specified trials
        mask = np.ones(len(success_data), dtype=bool)
        mask[exclude_trials] = False
        success_data = success_data[mask]
    
    return np.mean(success_data)

def get_timeseries_from_trial(nwb_file: NWBFile, trial_index: int, timeseries_path: str, start_label: str = "start_time", stop_label: str = "stop_time") -> np.ndarray:
    """
    Get the data from a trial.
    """
    start_time = nwb_file.trials[start_label][trial_index]
    stop_time = nwb_file.trials[stop_label][trial_index]
    timeseries = resolve_path(nwb_file, timeseries_path)
    start_idx = np.searchsorted(timeseries.timestamps[:], start_time)
    stop_idx = np.searchsorted(timeseries.timestamps[:], stop_time)
    return timeseries.data[start_idx:stop_idx]

def bitrate(nwb_file: NWBFile, timeseries_path: str, trial_start_label: str = "cue_time", trial_stop_label: str = "stop_time", target_label: str = "targets", target_radius_label: str = "target_radius", exclude_failed_trials: bool = True, exclude_intarget_trials: bool = True) -> float:
    """
    Calculate the throughput/bitrate using the formula:
    Throughput = Σₖ log₂(1 + (Dₖ-S)/2S) / t_acq
    
    Where:
    - Dₖ is the distance between initial value and target for dimension k
    - S is the target radius
    - t_acq is the total trial time
    
    Parameters:
    -----------
    nwb_file : NWBFile
        The NWB file containing trial data
    timeseries_path : str
        Path to the timeseries data
    trial_start_label : str, optional
        Label for trial start time (default: "cue_time")
    trial_stop_label : str, optional  
        Label for trial stop time (default: "stop_time")
    target_label : str, optional
        Label for target data (default: "target")
    target_radius_label : str, optional
        Label for target radius data (default: "target_radius")
    exclude_failed_trials : bool, optional
        Whether to exclude failed trials from the calculation (default: True)
    exclude_intarget_trials : bool, optional
        Whether to exclude trials where the initial value is within the target radius from the target (default: True)
    Returns:
    --------
    float
        The average throughput across all trials
    """
    trial_throughputs = []
    
    for trial_index in range(len(nwb_file.trials)):
        trial_start_time = nwb_file.trials[trial_start_label][trial_index]
        trial_stop_time = nwb_file.trials[trial_stop_label][trial_index]
        trial_duration = trial_stop_time - trial_start_time - nwb_file.trials['hold_time'][trial_index]
        target = nwb_file.trials[target_label][trial_index]
        target_radius = nwb_file.trials[target_radius_label][trial_index]  # S
        if exclude_failed_trials:
            if not nwb_file.trials['success'][trial_index]:
                trial_throughputs.append(np.nan)
                continue
        timeseries = get_timeseries_from_trial(nwb_file, trial_index, timeseries_path, start_label=trial_start_label, stop_label=trial_stop_label)
        # Get initial value of the timeseries
        # Add extra index if ndim ==1
        if timeseries.ndim == 1:
            timeseries = timeseries[: , None]
        initial_value = timeseries[0, :]
        if exclude_intarget_trials:
            if np.all(np.abs(initial_value - target) <= target_radius):
                trial_throughputs.append(np.nan)
                continue
        
        # Calculate Dₖ for each dimension k (distance between initial value and target)
        distances = np.abs(initial_value - target)  # Dₖ
        
        # Calculate the sum: Σₖ log₂(1 + (Dₖ-S)/2S)
        # Handle potential division by zero or negative arguments to log
        log_terms = []
        for d_k in distances:
            dk_minus_s = d_k - target_radius if d_k - target_radius > 0 else 0
            argument = 1 + dk_minus_s / (2 * target_radius)
            if argument > 0:
                log_terms.append(np.log2(argument))
            else:
                # If argument <= 0, use a small positive value to avoid log(0) or log(negative)
                log_terms.append(np.log2(1e-10))
        
        sum_log_terms = np.sum(log_terms)
        
        # Calculate throughput for this trial: Σₖ log₂(1 + (Dₖ-S)/2S) / t_acq
        trial_throughput = sum_log_terms / trial_duration
        trial_throughputs.append(trial_throughput)
    
    # Return average throughput across all trials
    return trial_throughputs