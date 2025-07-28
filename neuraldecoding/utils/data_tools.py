import numpy as np
import torch
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler
from neuraldecoding.dataaugmentation import SequenceScaler
import pandas as pd
import glob
import os
import re
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from datetime import datetime
import pickle

def extract_dates_from_filenames(data_path):
    # Find all matching .pkl files
    pkl_files = glob.glob(os.path.join(data_path, '*_preprocess.pkl'))

    dates = []
    for file_path in pkl_files:
        filename = os.path.basename(file_path)
        match = re.match(r'(\d{4}-\d{2}-\d{2})_preprocess\.pkl', filename)
        if match:
            dates.append(match.group(1))

    dates = np.asarray([datetime.strptime(date, '%Y-%m-%d') for date in dates])
    return dates #sorted(dates) 

def load_day(date, data_path):
        file = os.path.join(data_path, f'{date.strftime("%Y-%m-%d")}_preprocess.pkl')

        with open(file, 'rb') as f:
            data_CO, data_RD = pickle.load(f)
        
        return data_CO, data_RD

def load_one_nwb(fp: str) -> dict:
    with NWBHDF5IO(fp, "r", load_namespaces=True) as io:
        nwb = io.read()
        ana = nwb.analysis  # TimeSeries are stored only here

        # helper → fetch from /analysis or raise clear error
        def ts(name: str):
            try:
                return ana[name]
            except KeyError as e:
                raise KeyError(
                    f"TimeSeries '{name}' not in /analysis of '{fp}'"
                ) from e

        # --- time and kinematics ---
        t_sec = ts("index_position").timestamps[:]
        time_ms = (np.round(t_sec * 1000)).astype(np.int32)

        finger_kinematics = np.column_stack(
            [
                ts(n).data[:].ravel()
                for n in (
                    "index_position",
                    "mrs_position",
                    "index_velocity",
                    "mrs_velocity",
                )
            ]
        ).astype(np.float64)

        # --- neural features ---
        sbp = (ts("SpikingBandPower").data[:] / 0.25).astype(np.float64)
        tcfr = ts("ThresholdCrossings").data[:].astype(np.int16)

        # --- trial info ---
        tr = nwb.trials.to_dataframe()
        trial_number = tr["trial_number"].to_numpy()
        trial_count = tr["trial_count"].to_numpy()
        target_positions = tr[
            ["index_target_position", "mrs_target_position"]
        ].to_numpy()
        run_id = tr["run_id"].iloc[0]
        target_style = tr["target_style"]
        trial_timeout = tr["trial_timeout"]

        trial_index = np.searchsorted(t_sec, tr["start_time"].to_numpy()).astype(
            np.int32
        )

        return dict(
            trial_number=trial_number,
            trial_count=trial_count,
            target_positions=target_positions,
            trial_timeout=trial_timeout,
            time=time_ms,
            finger_kinematics=finger_kinematics,
            sbp=sbp,
            tcfr=tcfr,
            trial_index=trial_index,
            target_style=target_style[0],
            run_id=np.full_like(trial_number, run_id),
        )
    
def neural_finger_from_dict(dict, neural_type):
    neural = dict[neural_type]
    finger = dict['finger_kinematics']
    trial_idx = dict['trial_index']
    return (neural, finger), trial_idx

def data_split_direct(data_X, data_Y, ratio):
    total_len = len(data_X)
    len1 = int(total_len * ratio)

    split1_X = data_X[:len1, :]
    split1_Y = data_Y[:len1, :]
    split2_X = data_X[len1:, :]
    split2_Y = data_Y[len1:, :]

    split1 = (split1_X, split1_Y)
    split2 = (split2_X, split2_Y)

    return split1, split2

def data_split_trial(x, y, trial_idx, split_ratio=0.8, seed = 42):
    boundaries = np.concatenate([trial_idx, [len(x)]])
    n_trials = len(trial_idx)
    if seed<0:
        n_samples = x.shape[0]
        n_val = int(n_samples * (1.0-split_ratio))

        # Take the last n_val samples for validation
        train_neural_split = x[:-n_val]
        train_kinematics_split = y[:-n_val]
        val_neural = x[-n_val:]
        val_kinematics = y[-n_val:]
        return (train_neural_split, train_kinematics_split), (val_neural, val_kinematics)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_trials, generator=g)
    n_test = int(n_trials * (1.-split_ratio))
    
    test_trials = perm[:n_test]
    train_trials = perm[n_test:]
    
    train_mask = torch.zeros(len(x), dtype=torch.bool)
    test_mask = torch.zeros(len(x), dtype=torch.bool)
    
    for trial in train_trials:
        start, end = boundaries[trial], boundaries[trial + 1]
        train_mask[start:end] = True
        
    for trial in test_trials:
        start, end = boundaries[trial], boundaries[trial + 1]
        test_mask[start:end] = True
    
    return (x[train_mask],y[train_mask]), (x[test_mask],y[test_mask])

def add_history(neural_data, seq_len):
    """
    Add history to the neural data.
    neural_data is of shape (n_samples, n_channels)
    the output is of shape (n_samples, seq_len, n_channels)
    """
    Xtrain1 = torch.zeros((int(neural_data.shape[0]), int(neural_data.shape[1]), seq_len))
    Xtrain1[:, :, 0] = torch.from_numpy(neural_data)
    for k1 in range(seq_len - 1):
        k = k1 + 1
        Xtrain1[k:, :, k] = torch.from_numpy(neural_data[0:-k, :])

    # for RNNs, we want the last timestep to be the most recent data
    Xtrain1 = torch.flip(Xtrain1, (2,))

    #  (n_samples, n_channels, seq_len)
    return Xtrain1

def add_history_numpy(neural_data, seq_len):
    """
    Add history to the neural data.
    neural_data is of shape (n_samples, n_channels)
    the output is of shape (n_samples, seq_len, n_channels)
    """
    Xtrain1 = torch.zeros((int(neural_data.shape[0]), int(neural_data.shape[1]), seq_len))
    Xtrain1[:, :, 0] = torch.from_numpy(neural_data)
    for k1 in range(seq_len - 1):
        k = k1 + 1
        Xtrain1[k:, :, k] = torch.from_numpy(neural_data[0:-k, :])

    # for RNNs, we want the last timestep to be the most recent data
    Xtrain1 = torch.flip(Xtrain1, (2,))

    #  (n_samples, n_channels, seq_len)
    return Xtrain1.numpy()

def prep_data_and_split(data_dict, seq_len, num_train_trials, stabilization=None):
    trial_index = data_dict['trial_index']
    if len(trial_index) > num_train_trials:
        test_len = np.min((len(trial_index)-1, 399)) #TODO: split by trials as done in data_split_trial
        if stabilization is not None:
            neural = stabilization.fit(data_dict['sbp'])
            finger = data_dict['finger_kinematics']
        else:
            neural = data_dict['sbp']
            finger = data_dict['finger_kinematics']
        neural_training = neural[:trial_index[num_train_trials]]
        neural_testing = neural[trial_index[num_train_trials]:trial_index[test_len]]
        
        finger_training = finger[:trial_index[num_train_trials]]
        finger_testing = finger[trial_index[num_train_trials]:trial_index[test_len]]

        # add history
        if seq_len > 0:
            neural_training_hist = add_history(neural_training, seq_len)
            neural_testing_hist = add_history(neural_testing, seq_len)

            input_scaler = SequenceScaler()
            neural_training_scaled = input_scaler.fit_transform(neural_training_hist.numpy())
            output_scaler = StandardScaler()
            finger_training_scaled = output_scaler.fit_transform(finger_training)

            input_scaler = SequenceScaler()
            neural_testing_scaled = input_scaler.fit_transform(neural_testing_hist.numpy())
            output_scaler = StandardScaler()
            finger_testing_scaled = output_scaler.fit_transform(finger_testing)

            return torch.tensor(neural_training_scaled), torch.tensor(neural_testing_scaled), torch.tensor(finger_training_scaled), torch.tensor(finger_testing_scaled)
        else:
            return torch.tensor(neural_training), torch.tensor(neural_testing), torch.tensor(finger_training), torch.tensor(finger_testing)

    else:
        raise Exception('not enough trials')
    
def prep_data_decoder(data_dict, seq_len, stabilization=None):
    neural = data_dict['sbp']
    finger = data_dict['finger_kinematics']

    if stabilization is not None:
        neural = stabilization.extract_latent_space(neural)
    # add history
    if seq_len > 0:
        neural_hist = add_history(neural, seq_len)
        input_scaler = SequenceScaler()
        neural_scaled = input_scaler.fit_transform(neural_hist.numpy())
        output_scaler = StandardScaler()
        finger_scaled = output_scaler.fit_transform(finger)
        return torch.tensor(neural_scaled), torch.tensor(finger_scaled)
    else:
        return neural, torch.tensor(finger)
    
def add_hist(X, Y, hist=10):
    nNeu = X.shape[1]

    adjX = np.zeros((X.shape[0]-hist, nNeu, hist+1))
    for h in range(hist+1):
        adjX[:,:,h] = X[h:X.shape[0]-hist+h,:]
    adjY = Y[hist:,:]

    adjX = adjX.reshape(adjX.shape[0],-1)

    adjX = np.column_stack([adjX, np.ones(adjX.shape[0])])
    return adjX, adjY

def add_training_noise(x,
                       bias_neural_std=None,
                       noise_neural_std=None,
                       noise_neural_walk_std=None,
                       bias_allchans_neural_std=None,
                       device='cpu'):
		"""Function to add different types of noise to training input data to make models more robust.
		Identical to the methods in Willet 2021.
		Args:
			x (tensor):                     neural data of shape [batch_size x seq_len x num_chans]
			bias_neural_std (float):        std of bias noise
			noise_neural_std (float):       std of white noise
			noise_neural_walk_std (float):  std of random walk noise
			bias_allchans_neural_std (float): std of bias noise, bias is same across all channels
			device (device):                torch device (cpu or cuda)
		"""
		if bias_neural_std:
			# bias is constant across time (i.e. the 3 conv inputs), but different for each channel & batch
			biases = torch.normal(torch.zeros(x.shape[:2]), bias_neural_std).unsqueeze(2).repeat(1, 1, x.shape[2])
			x = x + biases.to(device=device)

		if noise_neural_std:
			# adds white noise to each channel and timepoint (independent)
			noise = torch.normal(torch.zeros_like(x), noise_neural_std)
			x = x + noise.to(device=device)

		if noise_neural_walk_std:
			# adds a random walk to each channel (noise is summed across time)
			noise = torch.normal(torch.zeros_like(x), noise_neural_walk_std).cumsum(dim=2)
			x = x + noise.to(device=device)

		if bias_allchans_neural_std:
			# bias is constant across time (i.e. the 3 conv inputs), and same for each channel
			biases = torch.normal(torch.zeros((x.shape[0], 1, 1)), bias_allchans_neural_std).repeat(1, x.shape[1], x.shape[2])
			x = x + biases.to(device=device)

		return x