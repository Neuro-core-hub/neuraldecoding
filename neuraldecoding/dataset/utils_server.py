import os
import warnings
from sys import platform
import neuraldecoding.dataset.utils_dataset as utils_dataset
import numpy as np
import numpy.matlib
import re
import copy
import glob
import warnings     
import pandas as pd

from ..utils import *

def get_server_data_path(is_monkey=True, custom_server_folder=None):
    """
        Returns the server filepath based on what operating system is being used.

        Inputs:
            is_monkey: bool
                If True, the path will be set to the monkey data folder. If False, the path will be set to the human data folder
            win_server_folder: str - Default: 'Z'
                The server folder path for Windows systems (if different from the default 'Z')
    """
    # choose the standard path based on the OS
    if custom_server_folder is not None:
        server_data_path = f'{custom_server_folder}/Data'
    elif platform == "linux" or platform == "linux2":
        server_data_path = '/run/user/1000/gvfs/smb-share:server=cnpl-drmanhattan.engin.umich.edu,share=share/Data'
    elif platform == "darwin":
        server_data_path = '/Volumes/share/Data'
    elif platform == "win32":
        server_data_path = f'Z:/Data'

    if is_monkey:
        server_data_path = os.path.join(server_data_path, 'Monkeys')
    else:
        server_data_path = os.path.join(server_data_path, 'Humans')

    return server_data_path

def get_server_notes_details(data_path):
    experiment_regex = '(?i)Experiment(er)?(s)?(:)?( )*'
    personnel_regex = '(?i)Personnel(:)?( )*'
    notes_regex = '(?i)Notes.*\.txt$'

    run_path_contents = os.listdir(data_path)

    for notes_file in run_path_contents:
        notes_file_path = os.path.join(data_path, notes_file)

        if re.search(notes_regex, notes_file) and os.path.isfile(notes_file_path):
            f = open(notes_file_path, 'r')
            notes_content = f.read()
            f.close()
            line_count = -1
            personnel_found_line = -1

            for line in notes_content.splitlines():
                experiment_found = re.search(experiment_regex, line)
                personnel_found = re.search(personnel_regex, line)

                if personnel_found:
                    experimenter = line[personnel_found.end():].replace(', ', ',').split(',')
                    personnel_found_line = line_count
                elif experiment_found and personnel_found_line == -1:
                    experimenter = line[experiment_found.end():].replace(', ', ',').split(',')

                line_count += 1

    return experimenter, notes_content

def load_z_script(direc):
    try:
        # load in and read z translator
        print(direc)
        f = open(r'{}'.format(os.path.join(direc, "zScript.txt")), "r")
        if f.mode == 'r':
            contents = f.read()
            f.close()

            return contents
    except:
        raise Exception("zScript.txt file not found. Make sure you're passing the right folder path.")

def read_xpc_data(contents, direc, num_channels, verbose=False):
    # supported data types and their byte sizes
    cls = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 2, 'uint32': 4, 'int32': 4, 'single': 4, 'double': 8}

    # data types and their python equivalent
    data_con = {'uint8': np.ubyte, 'int8': np.byte, 'uint16': np.ushort, 'int16': np.short, 'uint32': np.uintc,
                'int32': np.intc, 'single': np.single, 'double': np.double}

    # Split Z string into its P/M/D/N substrings:
    zstr = re.split(':.:', contents) #1st entry is blank

    # Split each substring into its fieldname/datatype/numels substrings:
    for i in range(len(zstr)):
        zstr[i] = zstr[i].split('-') #last entry in each cell is blank

    #Collect names, types, and sizes into list of list
    names = []
    types = []
    sizes = []
    for i in range(1, len(zstr)):
        names.append([])
        types.append([])
        sizes.append([])
        for j in range(0, len(zstr[i]) - 1, 3):
            names[i - 1].append(zstr[i][j])
        for j in range(1, len(zstr[i]) - 1, 3):
            types[i - 1].append(zstr[i][j])
        for j in range(2, len(zstr[i]) - 1, 3):
            sizes[i - 1].append(zstr[i][j])

    # Set flag(s) for specific field formatting:
    for i in range(len(names)):
        for j in range(len(names[i])):
            if names[i][j] == 'SpikeChans':
                spike_format = True
            else:
                spike_format = False

    #Recover number of fields in each file type:
    fnum = []
    for i in range(len(names)):
        fnum.append(len(names[i])) # Number of fields in each file

    fnames = [None] * 2 * sum(fnum)
    # use ord() to change hexidecimal notation to correct int values
    bsizes = copy.deepcopy(sizes)
    for i in range(len(fnum)):
        for j in range(len(names[i])):
            bsizes[i][j] = ord(bsizes[i][j])

    # Calculate byte sizes for each feature and collect field names:
    m = 0
    for i in range(len(fnum)):
        for j in range(len(names[i])):
            fnames[m] = names[i][j]
            m += 2
            # calculate the bytes sizes
            bsizes[i][j] = cls[types[i][j]] * bsizes[i][j] #Match type to cls, get type byte size, multiply by feature length:

    # Calculate bytes per timestep for each file:
    bytes = []
    for i in range(len(bsizes)):
        bytes.append(int(np.sum(bsizes[i]) + 2)) #plus 2 for each trial count

    # Get number of trials in this run:
    trial_list = glob.glob(os.path.join(direc, 'tParams*'))
    n_trials, trials = __find_num_trials(trial_list)

    # initalize the dictionary with correct field names
    dict_data = __set_field_names(fnames, trials, num_channels, spike_format)

    ########################## Parse Data Strings Into Dictionary: ##########################
    data = [[], [], [], []] #initilize data
    dropped_list = []
    for i in range(trials[-1]):
        try:
            # add trial number to dict
            dict_data[i]['TrialNumber'] = i + 1

            #read in data files
            data[0] = np.fromfile(os.path.join(direc, 'tParams{}.bin'.format(i + 1)), dtype='uint8')
            data[1] = np.fromfile(os.path.join(direc, 'mBehavior{}.bin'.format(i + 1)), dtype='uint8')
            data[2] = np.fromfile(os.path.join(direc, 'dBehavior{}.bin'.format(i + 1)), dtype='uint8')
            data[3] = np.fromfile(os.path.join(direc, 'neural{}.bin'.format(i + 1)), dtype='uint8')

        except:
            dict_data[i]['TrialNumber'] = None # this will set up the removal of empty dictionaries
            dropped_list.append(i) # sets up the spike formatting
            if verbose:
                print("Trail Number {} was dropped".format(i+1))
            continue # this skips to the next trial

        # Iterate through file types 1-3 and add data to Z:
        for j in range(4 - spike_format):

            # Calculate # of timesteps in this file:
            nstep = int(len(data[j]) / bytes[j])

            # Calculate the byte offsets for each feature in the timestep:
            offs = (3 + np.cumsum(bsizes[j])).tolist() #cumsum only works on np arrays so convert to list after
            offs.insert(0, 3) # starts at 3 because of trail counts

            # Iterate through each field:
            for k in range(fnum[j]):
                # Create a byte mask for the uint8 data:
                bmask = np.zeros(bytes[j], dtype=np.uint8)
                bmask[range(offs[k] - 1, offs[k] + bsizes[j][k] - 1)] = 1
                bmask = np.matlib.repmat(bmask, 1, nstep)
                bmask = bmask[0]

                #Extract data and cast to desired type:
                dat = data[j][bmask == 1].view((data_con[types[j][k]]))  # this has to be types

                #Reshape the data and add to dict
                dict_data[i][names[j][k]] = np.reshape(dat, (nstep, -1))

                #format the data so scalars are not in lists, arrays are in lists, and multiple arrays are lists of lists
                if len(dict_data[i][names[j][k]]) == 1: # if value is of form [[...]]  and not [[...][...][...][...]]
                    if len(dict_data[i][names[j][k]][0]) != 1:
                        dict_data[i][names[j][k]] = dict_data[i][names[j][k]][0]
                    else:
                        dict_data[i][names[j][k]] = dict_data[i][names[j][k]][0][0]

        # Extract Neural data packets (split around TrialCount and End Packet byte):
        if spike_format:
            new_string = int_to_string(data[3]) # convert to one continuous string
            try:
                # If it is one of these special characters python doesnt like it and understand that its not special
                # so add the \ to escape the special character
                # for some reason though | thinks its super special
                # if you just add the backslash it then grabs that when matching and messes up
                # so the second if statement is removing that character from each match the first time it appears
                # and that works but I have no idea why this is such a problem
                if chr((i+1)%256) in ['.', '*', '^', '$', '+', '?', '{', '}', '[', ']', '|', '(', ')' ,'\\']:
                    ndata = re.findall('\\' + int_to_string(np.asarray([np.ushort(i + 1)]).view(np.ubyte)) + '[^ÿ]*ÿ', new_string)
                else:
                    ndata = re.findall(int_to_string(np.asarray([np.ushort(i+1)]).view(np.ubyte)) + '[^ÿ]*ÿ', new_string)

                neural_data = []
                for m in range(len(ndata)):
                    if len(ndata[m][2:-1]) == 0:
                        neural_data.append([])
                    elif len(ndata[m][2:-1]) == 1:
                        neural_data.append([ord(ndata[m][2:-1])])
                    else:
                        neural_data_2 = []
                        for n in range(len(ndata[m][2:-1])):
                            neural_data_2.append(ord(ndata[m][2:-1][n]))
                        neural_data.append(neural_data_2)

                dict_data[i]['NeuralData'] = neural_data

            except re.error: # data is an empty cell
                dict_data[i]['NeuralData'] = []

    # Format Specific Fields
    ############################################################################
    # Change neural data field into spike times per channel:
    if spike_format:
        for i in range(trials[-1]):
            if i in dropped_list: # if i equals the value of a dropped trial then skip it
                continue

            spike_nums = np.zeros((num_channels, len(dict_data[i]['NeuralData'])))
            for t in range(len(dict_data[i]['NeuralData'])):
                for j in range(len(dict_data[i]['NeuralData'][t])):
                    if dict_data[i]['NeuralData'][t][j] != 0:
                        spike_nums[dict_data[i]['NeuralData'][t][j] - 1, t] += 1

            for c in range(num_channels):
                if np.any(spike_nums[c, :]):
                    times = dict_data[i]['ExperimentTime'][spike_nums[c, :] == 1]
                    spike_num_si = spike_nums[c, (spike_nums[c, :] == 1)]
                    idx = np.cumsum(spike_num_si)
                    j = np.ones((1, int(idx[-1])), dtype=int)

                    dict_data[i]['Channel'][c]['SpikeTimes'] = times[np.cumsum(j) - 1].T

                if len(dict_data[i]['Channel'][c]['SpikeTimes']) == 1:
                    if len(dict_data[i]['Channel'][c]['SpikeTimes'][0]) != 1:
                        dict_data[i]['Channel'][c]['SpikeTimes'] = dict_data[i]['Channel'][c]['SpikeTimes'][0].astype(int)
                    else:
                        dict_data[i]['Channel'][c]['SpikeTimes'] = dict_data[i]['Channel'][c]['SpikeTimes'][0][0].astype(int)

            #Removes these two fields
            dict_data[i].pop('SpikeChans', None)
            dict_data[i].pop('NeuralData', None)


    # this next one removes the skipped trials
    # must use a generator because of indexing issues if you don't
    if trials[-1] != n_trials: # only run if there is a trial that is dropped to save time
        dict_data = [trial for trial in dict_data if trial['TrialNumber'] != None]
    
    return dict_data

def __find_num_trials(trial_list):    
    """
    Find the number of trials and the trial numbers from the list of trial files
    """    
    ntrials = len(trial_list)
    trials = []
    for i in range(ntrials):
        trials.append(int(re.findall('\d+', trial_list[i])[-1]))
    trials = np.sort(trials)
    if trials[-1] != ntrials:
        warnings.warn("There is at least 1 dropped trial")

    return ntrials, trials

def __set_field_names(fnames, trials, num_channels, spike_format):
    dict_data = []

    for j in range(trials[-1]):  # this creates a dictionary for each trial
        all_data = {}

        for i in range(0, len(fnames), 2):
            all_data[fnames[i]] = None

        all_data['TrialNumber'] = None

        if spike_format: # this is so its a nested dictionary and can match other formats
            all_data['NeuralData'] = None
            all_data['Channel'] = []

            for k in range(num_channels):
                all_data['Channel'].append({'SpikeTimes': []})

        dict_data.append(all_data)

    return dict_data