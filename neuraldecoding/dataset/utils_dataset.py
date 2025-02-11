import os
import yaml
import numpy as np
from pynwb.file import Subject
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup, TimeSeries

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_creator_details():
    """
    Get the creator details for the NWB file

    Outputs:
        institution: str
            Institution name
        lab: str
            Lab name
    """

    # Load the creator_info YAML file
    with open(os.path.normpath(f'{DIR_PATH}/config/datasets_info.yaml'), 'r') as file:
        data = yaml.safe_load(file)

    institution = data['institution']
    lab = data['lab']

    return institution, lab

def get_device_and_electrode_group(data_type):
    """
    Get the Device object for the given data type

    Inputs:
        data_type: str
            Data type of the recording

    Outputs:
        device: pynwb.device.Device
            Device object
    """

    # Load the datasets_info YAML file
    with open(os.path.normpath(f'{DIR_PATH}/config/datasets_info.yaml'), 'r') as file:
        data = yaml.safe_load(file)

    if data_type == 'utah':
        device_info = data['devices']['BrainUtah']
    elif data_type == 'emg':
        device_info = data['devices']['MusclesEMG']
    else:
        raise ValueError(f"Data type '{data_type}' not supported")
    
    device = Device(
        name = device_info['name'], 
        description = device_info['description'],
        manufacturer = device_info['manufacturer'])
    
    electrode_group = ElectrodeGroup(
        name = device_info['group_name'],
        description = device_info['group_description'],
        device = device,
        location = device_info['group_location'])

    return device, electrode_group

def get_subject(subject_name):
    """
    Return the Subject NWB object for the given subject name

    Inputs:
        subject_name: str
            Name of the subject

    Outputs:
        subject: pynwb.file.Subject
            Subject object
    """
    # Load the subjects_ids YAML file
    with open(os.path.normpath(f'{DIR_PATH}/config/subjects_ids.yaml'), 'r') as file:
        data = yaml.safe_load(file)

    subject_ids = data['subject_ids']

    subject = Subject(description=subject_name, subject_id=subject_ids[subject_name])
    
    return subject

def get_dataset_variable_name(var_name):
    """
    Get the dataset variable name for the given variable name

    Inputs:
        var_name: str
            Variable name

    Outputs:
        dataset_var_name: str
            Dataset variable name
    """

    # Load the datasets_info YAML file
    with open(os.path.normpath(f'{DIR_PATH}/config/datasets_variable_names.yaml'), 'r') as file:
        dataset_variables = yaml.safe_load(file)

    if var_name not in dataset_variables:
        raise ValueError(f"Variable '{var_name}' not found in the dataset variable names config file")

    return dataset_variables[var_name]

def calc_bins_sbp(sbp_data, sample_width, bins_start_digit, bins_stop_digit):
    """
    Calculate the bins Spiking Band Power (SBP) for binned SBP data

    Inputs:
        sbp_data: np.array (n_times, n_channels)
            SBP data
        sample_width: np.array (n_times, 1)
            Sample width data
        bins_start_digit: np.array
            Bins start digitized
        bins_stop_digit: np.array
            Bins stop digitized

    Outputs:
        bins_sbp: np.array
            Bins SBP data
    """
    bins_sbp = np.zeros((len(bins_start_digit), sbp_data.shape[1]))

    for i in range(len(bins_start_digit)):
        start = bins_start_digit[i]
        stop = bins_stop_digit[i]

        indices_i = np.linspace(start, stop, abs(stop - start) + 1, dtype=int)
        bins_sbp[i] = np.sum(sbp_data[indices_i, :], axis=0) / np.sum(sample_width[indices_i])

    return bins_sbp