import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from neuraldecoding.utils import data_split_trial, load_one_nwb, eval_metrics
from neuraldecoding.model.Model import Model
from neuraldecoding.trainer.Trainer import Trainer
from neuraldecoding.model import linear_models as linear_models
from neuraldecoding.model.linear_models import LinearRegression, RidgeRegression, KalmanFilter
from neuraldecoding.dataset import Dataset
from neuraldecoding.preprocessing import Preprocessing
import os
import pickle
import matplotlib.pyplot as plt

class LinearTrainer(Trainer):
    def __init__(self, preprocessor: Preprocessing, config: DictConfig, dataset = None):
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
        model_class = getattr(linear_models, config.type)
        model = model_class(config.params)
        return model
    
    def train_model(self, plot_results = False):
        if self.cfg.model.params.get("is_refit", False):
            if self.cfg.model.params.get("prev_model_path", None) is None:
                raise ValueError("model.params.prev_model_path is not set in config. Necessary for refit training.")
            else:
                # Load the model first
                self.model.load_model(fpath=self.cfg.model.params.prev_model_path)
        self.model.train_step((self.train_X, self.train_Y))
        # Validate model
        self.validate_model(plot_results)
        # for metric in self.metrics:
        #     self.writer.add_scalar(f"{metric}/train", np.nanmean(self.logger[metric]['train'][-1]), 0)
        #     self.writer.add_scalar(f"{metric}/val", np.nanmean(self.logger[metric]['valid'][-1]), 0)
        print("Model trained, metrics:")
        self.save_print_log()
        return self.model, self.logger
    
    def validate_model(self, plot_results = False):
        train_prediction = self.model(self.train_X)
        valid_prediction = self.model(self.valid_X)
        for metric in self.metrics:
            metric_method = getattr(eval_metrics, metric)
            self.logger[metric]['train'].append(metric_method(train_prediction, self.train_Y))
            self.logger[metric]['valid'].append(metric_method(valid_prediction, self.valid_Y))
        
        # Show predictions vs targets
        # Get number of output dimensions
        num_outputs = valid_prediction.shape[-1] if len(valid_prediction.shape) > 1 else 1
        
        # Create subplots for each output dimension - validation
        fig_predictions, axes_predictions = plt.subplots(num_outputs, 1, figsize=(10, 4 * num_outputs))
        if num_outputs == 1:
            axes_predictions = [axes_predictions]
        
        for i in range(num_outputs):
            if len(valid_prediction.shape) > 1:
                pred_data = valid_prediction[:, i]
                target_data = self.valid_Y[:, i]
            else:
                pred_data = valid_prediction
                target_data = self.valid_Y
            
            axes_predictions[i].plot(target_data, label='Target', alpha=0.7)
            axes_predictions[i].plot(pred_data, label='Prediction', alpha=0.7)
            axes_predictions[i].set_xlabel('Sample')
            axes_predictions[i].set_ylabel(f'Output {i}')
            axes_predictions[i].set_title(f'Validation Predictions vs Targets - Dimension {i}')
            axes_predictions[i].legend()
            axes_predictions[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.writer.add_figure('validation/predictions_vs_targets', fig_predictions, global_step=0)
        
        # Create subplots for each output dimension - training
        fig_train_predictions, axes_train_predictions = plt.subplots(num_outputs, 1, figsize=(10, 4 * num_outputs))
        if num_outputs == 1:
            axes_train_predictions = [axes_train_predictions]
        
        for i in range(num_outputs):
            if len(train_prediction.shape) > 1:
                pred_data = train_prediction[:, i]
                target_data = self.train_Y[:, i]
            else:
                pred_data = train_prediction
                target_data = self.train_Y
            
            axes_train_predictions[i].plot(target_data, label='Target', alpha=0.7)
            axes_train_predictions[i].plot(pred_data, label='Prediction', alpha=0.7)
            axes_train_predictions[i].set_xlabel('Sample')
            axes_train_predictions[i].set_ylabel(f'Output {i}')
            axes_train_predictions[i].set_title(f'Train Predictions vs Targets - Dimension {i}')
            axes_train_predictions[i].legend()
            axes_train_predictions[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.writer.add_figure('train/predictions_vs_targets', fig_train_predictions, global_step=0)

