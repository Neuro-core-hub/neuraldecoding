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
from neuraldecoding.dataset import Dataset
from neuraldecoding.preprocessing import Preprocessing
import os
import pickle

class LinearTrainer(Trainer):
    def __init__(self, preprocessor: Preprocessing, config: DictConfig, dataset):
        super().__init__(config)
        self.model = self.create_model(self.cfg.model)
        self.preprocessor = preprocessor
        if dataset is not None:
            self.data_dict = self.load_data(dataset)
        #setting dummy num epochs to print stuff
        self.num_epochs = 1

    def load_data(self, dataset):
        result = self.preprocessor.preprocess_pipeline(dataset, params={'is_train': True})
        self.train_X = result['X_train']
        self.train_Y = result['Y_train']
        self.valid_X = result['X_val']
        self.valid_Y = result['Y_val']

    def create_model(self, config):
        """Creates and returns a loss function based on the configuration."""
        model_class = getattr(neuraldecoding.model.linear_models, config.type)
        model = model_class(config.params)
        return model
    
    def train_model(self):
        self.model.train_step((self.train_X, self.train_Y))
        # Validate model
        self.validate_model()
        print("Model trained, metrics:")
        self.save_print_log()
        return self.model, self.logger
    
    def validate_model(self):
        train_prediction = self.model(self.train_X)
        valid_prediction = self.model(self.valid_X)
        for metric in self.metrics:
            metric_method = getattr(neuraldecoding.utils.eval_metrics, metric)
            self.logger[metric][0].append(metric_method(train_prediction, self.train_Y))
            self.logger[metric][1].append(metric_method(valid_prediction, self.valid_Y))
    
        
