import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from neuraldecoding.utils import data_split_trial, load_one_nwb
from neuraldecoding.model.Model import Model
from neuraldecoding.trainer.Trainer import Trainer
import neuraldecoding.model.linear_models
from neuraldecoding.model.linear_models import LinearRegression, RidgeRegression, KalmanFilter
import os
import pickle

class LinearTrainer(Trainer):
    def __init__(self, preprocessor, config):
        super().__init__()
        self.model = self.create_model(config.model)
        self.preprocessor = preprocessor
        self.data_path = config.data.data_path
        self.train_X, self.train_Y = self.load_data()

    def load_data(self): # TODO, finalize this when dataset is merged to main
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        """Assuming data is dictionary output of one NWB file, change later"""
        data = load_one_nwb(self.data_path)
        # with open(self.data_path, "rb") as f:
        #     data = pickle.load(f)
        # (train_X, train_Y), (valid_X, valid_Y) = data_split_trial(data['sbp'], data['finger_kinematics'], data['trial_index'], self.split_ratio, self.split_seed)
        # return train_X, train_Y, valid_X, valid_Y
        train_X, _, train_Y,_ = self.preprocessor.preprocess_pipeline(data, params={'is_train': True})
        return train_X, train_Y
    
    def create_model(self, config):
        """Creates and returns a loss function based on the configuration."""
        model_class = getattr(neuraldecoding.model.linear_models, config.type)
        model = model_class(config.params)
        return model
    
    def train_model(self):
        self.model.train_step((self.train_X, self.train_Y))
        return self.model, None