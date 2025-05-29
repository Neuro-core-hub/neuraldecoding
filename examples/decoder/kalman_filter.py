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

# Example script of decoding data using ridge regression with decoder
# Data path needs to be specified first here:
data_path = "/mnt/D8C4D588C4D56970/ND/github/LINK_dataset/data/pickles"


# Load in config
config_path = os.path.join("configs","decoder","exampleKalmanFilter.yaml")
with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)
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

mses = []
corrs = []
for j in tqdm(range(finger_test.shape[0])):
    prediction = decoder.predict(neural_test[j,:].unsqueeze(0))
    print(prediction)
    mse = torch.mean((finger_test[j, :].unsqueeze(0) - prediction) ** 2)
    corr, _ = pearsonr(finger_test[j, :].numpy(), prediction[0, :].numpy())
    mses.append(mse.item())
    corrs.append(corr)

print("Average Mean Squared Error:", np.mean(mses))
print("Average Pearson Correlation:", np.mean(corrs))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(mses, label="Mean Squared Error", color="blue")
plt.xlabel("Test Sample Index")
plt.ylabel("MSE")
plt.title("Mean Squared Error per Test Sample")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(corrs, label="Pearson Correlation", color="green")
plt.xlabel("Test Sample Index")
plt.ylabel("Correlation")
plt.title("Pearson Correlation per Test Sample")
plt.legend()

plt.show()
