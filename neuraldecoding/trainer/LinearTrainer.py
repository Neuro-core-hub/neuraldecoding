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
        model_class = getattr(neuraldecoding.model.linear_models, config.type)
        model = model_class(config.params)
        return model
    
    def train_model(self, plot_results = False):
        self.model.train_step((self.train_X, self.train_Y))
        # Validate model
        self.validate_model(plot_results)
        print("Model trained, metrics:")
        self.save_print_log()
        return self.model, self.logger
    
    def validate_model(self, plot_results = False):
        train_prediction = self.model(self.train_X)
        valid_prediction = self.model(self.valid_X)
        for metric in self.metrics:
            metric_method = getattr(neuraldecoding.utils.eval_metrics, metric)
            self.logger[metric][0].append(metric_method(train_prediction, self.train_Y))
            self.logger[metric][1].append(metric_method(valid_prediction, self.valid_Y))
        if plot_results:
            # Create plots for training and validation predictions
            n_dofs = self.train_Y.shape[1] if len(self.train_Y.shape) > 1 else 1
            
            # Calculate optimal subplot layout for DoFs
            if n_dofs == 1:
                rows, cols = 1, 1
            elif n_dofs <= 4:
                rows, cols = 2, 2
            elif n_dofs <= 6:
                rows, cols = 2, 3
            elif n_dofs <= 9:
                rows, cols = 3, 3
            elif n_dofs <= 12:
                rows, cols = 3, 4
            else:
                rows = int(np.ceil(np.sqrt(n_dofs)))
                cols = int(np.ceil(n_dofs / rows))
            
            # Create training figure
            fig_train, axes_train = plt.subplots(rows, cols, constrained_layout=True, figsize=(cols*4, rows*3))
            fig_train.suptitle('Training Results')
            if n_dofs == 1:
                axes_train = [axes_train]
            else:
                axes_train = axes_train.flatten()
            
            # Create validation figure
            fig_val, axes_val = plt.subplots(rows, cols, constrained_layout=True, figsize=(cols*4, rows*3))
            fig_val.suptitle('Validation Results')
            if n_dofs == 1:
                axes_val = [axes_val]
            else:
                axes_val = axes_val.flatten()
            
            for dof in range(n_dofs):
                # Extract data for this DoF
                if n_dofs == 1:
                    train_actual = self.train_Y
                    train_pred = train_prediction
                    valid_actual = self.valid_Y
                    valid_pred = valid_prediction
                else:
                    train_actual = self.train_Y[:, dof]
                    train_pred = train_prediction[:, dof]
                    valid_actual = self.valid_Y[:, dof]
                    valid_pred = valid_prediction[:, dof]
                
                # Training plot
                axes_train[dof].plot(train_actual, label='Actual', alpha=0.7)
                axes_train[dof].plot(train_pred, label='Predicted', alpha=0.7)
                axes_train[dof].set_title(f'DoF {dof + 1}')
                axes_train[dof].set_xlabel('Time')
                axes_train[dof].set_ylabel('Value')
                axes_train[dof].legend()
                axes_train[dof].grid(True, alpha=0.3)
                
                # Validation plot
                axes_val[dof].plot(valid_actual, label='Actual', alpha=0.7)
                axes_val[dof].plot(valid_pred, label='Predicted', alpha=0.7)
                axes_val[dof].set_title(f'DoF {dof + 1}')
                axes_val[dof].set_xlabel('Time')
                axes_val[dof].set_ylabel('Value')
                axes_val[dof].legend()
                axes_val[dof].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_dofs, len(axes_train)):
                axes_train[i].set_visible(False)
                axes_val[i].set_visible(False)
            
            plt.show()
