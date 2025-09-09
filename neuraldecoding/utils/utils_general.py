import os
import sys
import numpy as np
from omegaconf import open_dict


def int_to_string(in_array):
    """
    Convert an array of integers to a string

    Inputs:
        in_array: list
            List of integers
    """
    final_string = []
    for number in range(len(in_array)):
        final_string.append(chr(in_array[number]))

    final_string = "".join(final_string)

    return final_string


def get_creation_path_time(path):
    """
    Get the creation time of the earliest file in the given directory path

    Inputs:
        path: str
            Path to the directory
    """
    earliest_time = sys.float_info.max

    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        # choose the standard path based on the OS
        if (
            sys.platform == "linux"
            or sys.platform == "linux2"
            or sys.platform == "win32"
        ):
            curr_time = os.path.getctime(file_path)
        elif sys.platform == "darwin":
            curr_time = os.stat(file_path).st_birthtime

        if curr_time < earliest_time:
            earliest_time = curr_time

    return earliest_time


def is_collection(obj):
    """
    Check if the object is a collection (list, tuple, or numpy array)
    """
    return isinstance(obj, (list, tuple, np.ndarray))

def resolve_path(obj, path):
    """
    given a string that resembles a class structure, access the attribute from obj
    ex: obj has attribute 'processing' that has attribute 'ecephys', path is 'processing.ecephys'
    this function will return obj.processing.ecephys

    Parameters:
    obj: the object we would like to access
    str: the path leading to the part of the object we'd like to access (not including the object itself)
    """
    for key in path.split('.'):
        try:
            obj = getattr(obj,key)
        except AttributeError:
            try:
                obj = obj[key]
            except (KeyError, TypeError):
                raise ValueError(f"Cannot resolve path segment: {key}")
    return obj

def export_preprocess_params(model_config, preprocess_config):
    for path in preprocess_config.model_conf_append:
        try:
            # Split the path to get the final key
            keys = path.split('.')
            final_key = keys[-1]
            # Get the value at the path
            value = resolve_path(preprocess_config, path)
            # Save the value to model_config.params with the final key
            with open_dict(model_config.params):
                model_config.params[final_key] = value
        except ValueError as e:
            print(f"Warning: {e}. Skipping path: {path}")
    return model_config