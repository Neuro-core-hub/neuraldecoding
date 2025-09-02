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


def data_split_trial(x, y, trial_idx=None, split_ratio=0.8, seed=42, shuffle=False):
    """
    Split (x, y) into train/test.
    - If trial_idx is provided and non-empty, split by whole trials.
      trial_idx is interpreted as the start index of each trial (sorted or not).
    - If trial_idx is None or empty, split by samples.
    - Shuffles only when shuffle=True (seeded).
    Returns: (x_train, y_train), (x_test, y_test)
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length.")

    g = torch.Generator().manual_seed(seed)
    device = x.device if torch.is_tensor(x) else torch.device('cpu')

    # ----- Trial-aware split -----
    if trial_idx is not None and len(trial_idx) > 0:
        # Normalize/validate trial starts
        if torch.is_tensor(trial_idx):
            tstarts = trial_idx.flatten().long().cpu()
        else:
            tstarts = torch.as_tensor(trial_idx, dtype=torch.long)
        tstarts = tstarts.clamp(0, max(0, n - 1))
        tstarts, _ = torch.sort(tstarts)

        # Build boundaries: [t0, t1, ..., tn, N]
        boundaries = torch.cat([tstarts, torch.tensor([n], dtype=torch.long)])

        n_trials = len(tstarts)
        n_train_trials = int(n_trials * split_ratio)

        if shuffle:
            perm = torch.randperm(n_trials, generator=g)
        else:
            perm = torch.arange(n_trials)

        train_trials = perm[:n_train_trials]
        test_trials = perm[n_train_trials:]

        # Boolean masks on the same device as x (if tensor)
        train_mask = torch.zeros(n, dtype=torch.bool, device=device)
        test_mask = torch.zeros(n, dtype=torch.bool, device=device)

        for tr in train_trials.tolist():
            s = int(boundaries[tr].item())
            e = int(boundaries[tr + 1].item())
            train_mask[s:e] = True

        for tr in test_trials.tolist():
            s = int(boundaries[tr].item())
            e = int(boundaries[tr + 1].item())
            test_mask[s:e] = True

        return (x[train_mask], y[train_mask]), (x[test_mask], y[test_mask])

    # ----- Sample-wise split (no trial info) -----
    n_train = int(n * split_ratio)
    if shuffle:
        perm = torch.randperm(n, generator=g).to(device)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
    else:
        train_idx = torch.arange(0, n_train, device=device)
        test_idx = torch.arange(n_train, n, device=device)

    return (x[train_idx], y[train_idx]), (x[test_idx], y[test_idx])


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
    return adjX, adjY