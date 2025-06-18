import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from neuraldecoding.utils import data_tools
import numpy as np
import torch
from scipy.stats import pearsonr
from neuraldecoding.model.linear_models import KalmanFilter
from neuraldecoding.decoder import LinearDecoder
import yaml
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from neuraldecoding.utils import parse_verify_config
from hydra import initialize, compose

# Example script of decoding data using ridge regression with decoder
# Data path needs to be specified first here:
# data_path = "/mnt/D8C4D588C4D56970/ND/github/LINK_dataset/data/pickles"
data_path = "D:\\ND\\github\\LINK_dataset\\data\\pickles"

# Load in config
cfg_path = os.path.join("..","..","configs","example_Kalman_Filter")
with initialize(version_base=None, config_path=cfg_path):
    config = compose("config")
cfg = parse_verify_config(config, 'decoder')

# Load in data
dates = data_tools.extract_dates_from_filenames(data_path)
date = dates[42]
data_CO, data_RD = data_tools.load_day(date, data_path)

# Data preprocessing
if data_CO and data_RD:
    TrialIndex = max(data_CO['trial_index'], data_RD['trial_index'])
    sbp = np.concatenate((data_CO['sbp'], data_RD['sbp']), axis=0)
    beh = np.concatenate((data_CO['finger_kinematics'], data_RD['finger_kinematics']), axis=0)
elif data_RD:
    TrialIndex = data_RD['trial_index']
    sbp = data_RD['sbp']
    beh = data_RD['finger_kinematics']
else:
    TrialIndex = data_CO['trial_index']
    sbp = data_CO['sbp']
    beh = data_CO['finger_kinematics']

# Train-test split
test_len = np.min((len(TrialIndex)-1, 399))
neural_train= sbp[:TrialIndex[300]]
neural_test = sbp[TrialIndex[300]:TrialIndex[test_len]]
finger_train = beh[:TrialIndex[300]]
finger_test = beh[TrialIndex[300]:TrialIndex[test_len]]

# Add history
def add_hist(X, Y, hist=10):
    nNeu = X.shape[1]

    adjX = np.zeros((X.shape[0]-hist, nNeu, hist+1))
    for h in range(hist+1):
        adjX[:,:,h] = X[h:X.shape[0]-hist+h,:]
    adjY = Y[hist:,:]

    adjX = adjX.reshape(adjX.shape[0],-1)
    return adjX, adjY

# neural_train, finger_train = add_hist(neural_train, finger_train, hist = 7)
# neural_test, finger_test = add_hist(neural_test, finger_test, hist = 7)

# Add bias term
# neural_test = np.concatenate((neural_test, np.ones((len(neural_test), 1))), axis=1)
# neural_train = np.concatenate((neural_train, np.ones((len(neural_train), 1))), axis=1)
neural_train = torch.tensor(neural_train, dtype=torch.float32)
neural_test = torch.tensor(neural_test, dtype=torch.float32)
finger_train = torch.tensor(finger_train, dtype=torch.float32)
finger_test = torch.tensor(finger_test, dtype=torch.float32)

# Train model using model module
model_path = cfg["fpath"]
model = KalmanFilter(cfg["model"]["parameters"])
model.train_step((neural_train, finger_train))
model.save_model(model_path)

# Load and evaluate model using decoder module
decoder = LinearDecoder(cfg)
decoder.load_model()

print("Decoding with Offline Setting")
predictions_offline = decoder.predict(neural_test)

print("Decoding with Online Setting")
predictions_online = torch.zeros_like(finger_test)
for j in tqdm(range(finger_test.shape[0])):
    if j == 500 or j == 1000 or j == 1500:
        # Example intervention
        print(f"Intervening initialization at index {j}")
        new_yhat = np.expand_dims(np.array([ 7.6127e-01,  4.8506e-01, -2.6989e-04, -5.7489e-05]),axis=0)
        decoder.model.initialize(new_yhat)

    prediction = decoder.predict(neural_test[j,:].unsqueeze(0))
    predictions_online[j, :] = prediction

corr_offline, _ = pearsonr(predictions_offline, finger_test)
print(f"Offline correlation: {corr_offline}")

corr_online, _ = pearsonr(predictions_online, finger_test)
print(f"Online correlation: {corr_online}")

plt.figure(figsize=(10, 6))
plt.plot(predictions_online.detach().numpy()[:,0], label="Offline Predictions", alpha=0.7)
plt.plot(predictions_offline.detach().numpy()[:,0], label="Online Predictions", alpha=0.7)
plt.plot(finger_test.detach().numpy()[:,0], label="Ground Truth", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Predictions vs Finger Test")
plt.legend()
plt.show()