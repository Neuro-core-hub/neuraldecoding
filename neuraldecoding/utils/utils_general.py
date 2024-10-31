import os
import sys

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

    final_string = ''.join(final_string)

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
        if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "win32":
            curr_time = os.path.getctime(file_path)
        elif sys.platform == "darwin":
            curr_time = os.stat(file_path).st_birthtime
        
        if curr_time < earliest_time:
            earliest_time = curr_time
    
    return earliest_time