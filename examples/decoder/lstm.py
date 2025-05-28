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
from neuraldecoding.trainer.NeuralNetworkTrainer import TrainerImplementation

#prep

def load_synthetic_data():
    """Loads the synthetic train and validation data from .npy files."""
    data_train = np.load(os.path.join("tests", "trainer", "data", "train.npz"), allow_pickle=True)
    train_X, train_Y = data_train["X"], data_train["Y"]
    
    data_valid = np.load(os.path.join("tests", "trainer", "data", "valid.npz"), allow_pickle=True)
    valid_X, valid_Y = data_valid["X"], data_valid["Y"]
    
    return train_X, train_Y, valid_X, valid_Y

def create_dataloaders(train_X, train_Y, valid_X, valid_Y, batch_size=32):
    """Creates PyTorch DataLoaders for training and validation data."""
    train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), 
                                  torch.tensor(train_Y, dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(valid_X, dtype=torch.float32), 
                                  torch.tensor(valid_Y, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

def load_config():
    """Loads the Hydra configuration file (train.yaml)."""
    return OmegaConf.load(os.path.join("configs", "trainer", "testTrainer.yaml"))

# load data and trainer config

train_X, train_Y, valid_X, valid_Y = load_synthetic_data()
train_loader, valid_loader = create_dataloaders(train_X, train_Y, valid_X, valid_Y)
trainer_config = load_config()
trainer = TrainerImplementation()

config_path = os.path.join("configs","decoder","exampleLSTM.yaml")
with open(config_path, "r") as file:
            decoder_config = yaml.safe_load(file)


model, results = trainer.train_model(train_loader, valid_loader, trainer_config)
model_path = decoder_config["fpath"]
model.save_model(decoder_config["fpath"])

# load decoder config
decoder = NeuralNetworkDecoder(decoder_config)
decoder.load_model()
valid_data_X = torch.tensor(valid_X, dtype=torch.float32)
valid_data_Y = torch.tensor(valid_Y, dtype=torch.float32)
lstm_prediction,_ = decoder.predict(valid_data_X)
print(lstm_prediction)

# eval
lstm_prediction_np = lstm_prediction.detach().numpy()
valid_data_Y_np = valid_data_Y.detach().numpy()

correlation, _ = pearsonr(lstm_prediction_np.flatten(), valid_data_Y_np.flatten())
print(f"Pearson correlation coefficient: {correlation}")
