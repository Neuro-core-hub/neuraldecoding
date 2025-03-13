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
        f = open(r'{}'.format(os.path.join(direc, "zScript.txt")), "r")
        if f.mode == 'r':
            contents = f.read()
            f.close()

            return contents
    except:
        raise Exception("zScript.txt file not found. Make sure you're passing the right folder path.")

# TODO: optimize this function - it's very slow right now

def read_xpc_data(contents, direc, num_channels, verbose=False):
    # supported data types and their byte sizes #START REMOVE
    print("Running optimized code")
    cls = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 2, 'uint32': 4, 'int32': 4, 'single': 4, 'double': 8}
    # data types and their python equivalent
    data_con = {'uint8': np.ubyte, 'int8': np.byte, 'uint16': np.ushort, 'int16': np.short, 'uint32': np.uintc,
                'int32': np.intc, 'single': np.single, 'double': np.double}

    # Split Z string into its P/M/D/N substrings:
    zstr = re.split(':.:', contents) #1st entry is blank

    # Split each substring into its fieldname/datatype/numels substrings:
    for i in range(len(zstr)):
        zstr[i] = zstr[i].split('-') #last entry in each cell is blank
    
    # Extract names, types, and sizes using list comprehension, starting from index 1
    names = [[zstr[i][j] for j in range(0, len(zstr[i])-1, 3)] for i in range(1, len(zstr))]
    types = [[zstr[i][j] for j in range(1, len(zstr[i])-1, 3)] for i in range(1, len(zstr))]
    sizes = [[zstr[i][j] for j in range(2, len(zstr[i])-1, 3)] for i in range(1, len(zstr))]

    #######-----------------
    #Collect names, types, and sizes into list of list
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
    spikeformat = any('SpikeChans' in field_list for field_list in names)

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
                raise ValueError(f"Unknown data type '{types[i][j]}' in cls dictionary.") from e

    # Calculate bytes per timestep for each file:
    bytes = [int(np.sum(file_bytes) + 2) for file_bytes in bsizes]

    # Get number of trials in this run
    trial_list = glob.glob(os.path.join(direc, 'tParams*'))
    ntrials = len(trial_list)

    # Extract and sort trial numbers more efficiently
    trials = sorted([int(re.findall(r'\d+', trial_path)[-1]) for trial_path in trial_list])

    # Check for dropped trials
    if trials[-1] != ntrials:
        warnings.warn("There is at least 1 dropped trial")

    # Initialize the dictionary with correct field names
    channel_structure = ([{'SpikeTimes': []} for _ in range(num_channels)] 
                        if spikeformat else None)

    base_dict = {
    **{name: None for name in fnames[::2]},
    'TrialNumber': None,
    **(
        {'NeuralData': [], 'Channel': channel_structure} 
        if spikeformat else {}
    )
    }

    # Create list of dictionaries for all trials
    dict_data = [copy.deepcopy(base_dict) for _ in range(trials[-1])]

############################## Parse Data Strings Into Dictionary: ######################################
    data = [[], [], [], []] #initilize data
    dropped_list = []
    file_templates = [
    'tParams{}.bin',
    'mBehavior{}.bin',
    'dBehavior{}.bin',
    'neural{}.bin'
    ]
    for i in range(trials[-1]):
        trial_num = i + 1
        dict_data[i]['TrialNumber'] = trial_num
        try:
            # add trial number to dict
            dict_data[i]['TrialNumber'] = i + 1

            #read in data files
            for file_idx, template in enumerate(file_templates):
                filepath = os.path.join(direc, template.format(trial_num))
                data[file_idx] = np.fromfile(filepath, dtype='uint8')

        except:
            dict_data[i]['TrialNumber'] = None # this will set up the removal of empty dictionaries
            dropped_list.append(i) # sets up the spike formatting
            if verbose:
                print("Trail Number {} was dropped".format(i+1))
            continue # this skips to the next trial

        # Iterate through file types 1-3 and add data to Z:
        for j in range(4 - spikeformat):

            # Calculate # of timesteps in this file:
            nstep = len(data[j]) // bytes[j]

            # Calculate the byte offsets for each feature in the timestep:
            offs = np.concatenate(([3], 3 + np.cumsum(bsizes[j])))

            # Iterate through each field:
            for k in range(fnum[j]):
                # Create a byte mask for the uint8 data:
                #----------xxxxxxxxxxxxxxxxxxx
                #mine
                mask_length = offs[k + 1] - offs[k]
                bmask = np.zeros(bytes[j], dtype=np.uint8)
                bmask[offs[k] - 1:offs[k] + mask_length - 1] = 1
                bmask = np.tile(bmask, nstep)
                #-------------xxxxxxxxxxxxxxxxxxxxxx
                
                
                

                

                #---------------xxxxxxxxxxxxxxxxx-------------------
                #Another try
                #Extract and reshape data in one step
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
                        dict_data[i][names[j][k]] = first_element  # Extract single array
                #--------------xxxxxxx-----------------

                

        
        # Extract Neural data packets (optimized for speed)
        if spikeformat:
            new_string = int_to_string(data[3])  # convert to one continuous string
            
            # Pre-compute the search pattern only once
            search_char = chr((i+1) % 256)
            if search_char in '.^$*+?{}[]|()\\':
                pattern = '\\' + int_to_string(np.asarray([np.ushort(i + 1)]).view(np.ubyte))
            else:
                pattern = int_to_string(np.asarray([np.ushort(i + 1)]).view(np.ubyte))
            
            # Single regex compilation and search
            pattern += '[^ÿ]*ÿ'
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
            


            except re.error: # data is an empty cell
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
            exp_time = dict_data[i]['ExperimentTime']
            neural_data_len = len(neural_data)
            
            # Use numpy for faster operations
            spikenums = np.zeros((num_channels, neural_data_len), dtype=np.int32)
            
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
            channel_data = dict_data[i]['Channel']
            for c in range(num_channels):
                # Only process channels with spikes
                spike_mask = spikenums[c, :] == 1
                if np.any(spike_mask):
                    # Get spike times directly with the mask
                    times = exp_time[spike_mask]
                    
                    # Simplify the counting logic
                    spike_count = np.sum(spike_mask)
                    
                    # Directly assign the spike times (transposed)
                    channel_data[c]['SpikeTimes'] = times.T
                    
                    # Format optimization - handle scalar vs array cases
                    spike_times = channel_data[c]['SpikeTimes']
                    if len(spike_times) == 1:
                        if len(spike_times[0]) != 1:
                            channel_data[c]['SpikeTimes'] = spike_times[0].astype(np.int32)
                        else:
                            channel_data[c]['SpikeTimes'] = int(spike_times[0][0])  # Direct int conversion
            
            # Remove unneeded fields (moved outside the channel loop)
            dict_data[i].pop('SpikeChans', None)
            dict_data[i].pop('NeuralData', None)
   

    # this next one removes the skipped trials
    # must use a generator because of indexing issues if you don't
    if trials[-1] != ntrials: # only run if there is a trial that is dropped to save time
        dict_data = [trial for trial in dict_data if trial['TrialNumber'] != None]

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
