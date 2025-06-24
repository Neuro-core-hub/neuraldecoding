import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from neuraldecoding.utils import data_tools
import numpy as np
import torch
from scipy.stats import pearsonr
from neuraldecoding.model.linear_models import LinearRegression, RidgeRegression
from neuraldecoding.decoder import NeuralNetworkDecoder
import yaml
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from neuraldecoding.trainer.NeuralNetworkTrainer import LSTMTrainer
from neuraldecoding.utils import parse_verify_config
from hydra import initialize, compose

#prep
cfg_path = os.path.join("..","..","configs","example_LSTM")

with initialize(version_base=None, config_path=cfg_path):
    config = compose("config")

def load_config():
    """Loads the Hydra configuration file (train.yaml)."""
    cfg = parse_verify_config(config, 'trainer')
    return cfg


# load data and trainer config

trainer_config = load_config()
trainer = LSTMTrainer(trainer_config)


decoder_config = parse_verify_config(config, 'decoder')

model, results = trainer.train_model()
model_path = decoder_config["fpath"]
model.save_model(decoder_config["fpath"])

# load decoder config
decoder = NeuralNetworkDecoder(decoder_config)
decoder.load_model()
valid_data_X = torch.tensor(trainer.valid_X, dtype=torch.float32)
valid_data_Y = torch.tensor(trainer.valid_Y, dtype=torch.float32)
lstm_prediction,_ = decoder.predict(valid_data_X)
print(lstm_prediction)

# eval
lstm_prediction_np = lstm_prediction.detach().numpy()
valid_data_Y_np = valid_data_Y.detach().numpy()

correlation, _ = pearsonr(lstm_prediction_np.flatten(), valid_data_Y_np.flatten())
print(f"Pearson correlation coefficient: {correlation}")
