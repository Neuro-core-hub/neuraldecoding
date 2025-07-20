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
    def __init__(self, dataset: Dataset, preprocessor: Preprocessing, config: DictConfig):
        super().__init__(config)
        self.model = self.create_model(self.cfg.model)
        self.preprocessor = preprocessor
        self.dataset = dataset
        self.logger = {metric: [[], []] for metric in self.cfg.metrics}

    def load_data(self):
        result = self.preprocessor.preprocess_pipeline(self.dataset, params={'is_train': True})
        self.train_X = result['neural_train']
        self.train_Y = result['behavior_train']
        self.valid_X = result['neural_test']
        self.valid_Y = result['behavior_test']
        
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
        self.print_metrics()
        return self.model, self.logger
    
    def validate_model(self):
        train_prediction = self.model(self.train_X)
        valid_prediction = self.model(self.valid_X)
        for metric in self.cfg.metrics:
            metric_method = getattr(neuraldecoding.utils.eval_metrics, metric)
            self.logger[metric][0].append(metric_method(train_prediction, self.train_Y))
            self.logger[metric][1].append(metric_method(valid_prediction, self.valid_Y))
    
        
