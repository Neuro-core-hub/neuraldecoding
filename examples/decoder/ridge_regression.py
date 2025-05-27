import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from neuraldecoding.utils import data_tools
import numpy as np
from scipy.stats import pearsonr
from neuraldecoding.model.linear_models import LinearRegression, RidgeRegression
from neuraldecoding.decoder.OfflineDecoders import LinearDecoder
import yaml
from sklearn.metrics import r2_score

# Example script of decoding data using ridge regression with decoder
# Data path needs to be specified first here:
data_path = "/mnt/D8C4D588C4D56970/ND/github/LINK_dataset/data/pickles"


# Load in config
config_path = os.path.join("configs","decoder","exampleRidgeRegression.yaml")
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

neural_train, finger_train = add_hist(neural_train, finger_train, hist = 7)
neural_test, finger_test = add_hist(neural_test, finger_test, hist = 7)

# Add bias term
neural_test = np.concatenate((neural_test, np.ones((len(neural_test), 1))), axis=1)
neural_train = np.concatenate((neural_train, np.ones((len(neural_train), 1))), axis=1)

# Train model using model module
model_path = cfg["fpath"]
model = RidgeRegression(params=cfg["model"]["parameters"])
model.train_step((neural_train, finger_train))
model.save_model(model_path)

# Load and evaluate model using decoder module
decoder = LinearDecoder(cfg)
decoder.load_model()
rr_prediction = decoder.predict(neural_test)

r = np.array([pearsonr(finger_test[:, i], rr_prediction[:, i])[0] for i in range(finger_test.shape[1])])
r_squared = [r2_score(finger_test[:, i], rr_prediction[:, i]) for i in range(finger_test.shape[1])]
print("Pearson r for each output dimension:", r)
print("r^2 for each output dimension:", r_squared)

