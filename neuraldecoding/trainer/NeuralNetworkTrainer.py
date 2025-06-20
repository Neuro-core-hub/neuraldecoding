import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from neuraldecoding.utils import prep_data_and_split, load_one_nwb
from neuraldecoding.model.Model import Model
from neuraldecoding.trainer.Trainer import Trainer
from neuraldecoding.model.neural_network_models.NeuralNetworkModel import NeuralNetworkModel
from neuraldecoding.model.neural_network_models.LSTM import LSTM
import os
class NNTrainer(Trainer):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.training.device)
        self.model = self.create_model(config.model).to(self.device)
        self.optimizer = self.create_optimizer(config.optimizer, self.model.parameters())
        self.scheduler = self.create_scheduler(config.scheduler, self.optimizer)
        self.loss_func = self.create_loss_function(config.loss_func)
        self.num_epochs = config.training.num_epochs
        self.batch_size = config.training.batch_size
        self.print_results = config.training.get("print_results", True)
        self.print_every = config.training.get("print_every", 10)
        self.data_path = config.data.data_path

    def load_data(self):
        pass

    def create_dataloaders(self):
        pass
    
    def create_optimizer(self, optimizer_config: DictConfig, model_params) -> Optimizer:
        """Creates and returns an optimizer based on the configuration."""
        optimizer_class = getattr(torch.optim, optimizer_config.type)
        return optimizer_class(model_params, **optimizer_config.params)

    def create_scheduler(self, scheduler_config: DictConfig, optimizer: Optimizer) -> _LRScheduler:
        """Creates and returns a learning rate scheduler based on the configuration."""
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config.type)
        return scheduler_class(optimizer, **scheduler_config.params)

    def create_loss_function(self, loss_config: DictConfig) -> torch.nn.Module:
        """Creates and returns a loss function based on the configuration."""
        loss_class = getattr(torch.nn, loss_config.type)
        return loss_class(**loss_config.params)
    
    def create_model(self, model_config: DictConfig) -> torch.nn.Module:
        """Creates and returns a loss function based on the configuration."""
        model_class = globals()[model_config['type']]  
        model = model_class(model_config['parameters']) 
        return model
    
    def calc_corr(self, y1,y2):
        """Calculates the correlation between y1 and y2 (tensors)"""
        corr = []
        for i in range(y1.shape[1]):
            corr.append(np.corrcoef(y1[:, i], y2[:, i])[1, 0])
        return corr

    def train_model(self, train_loader = None, valid_loader = None):
        pass


class LSTMTrainer(NNTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.split_ratio = config.data.params.split_ratio
        self.split_seed = config.data.params.split_seed
        self.data_path = config.data.data_path
        self.sequence_length = config.data.params.sequence_length
        self.num_train_trials = config.data.params.num_train_trials
        self.train_X, self.train_Y, self.valid_X, self.valid_Y = self.load_data()
        self.train_loader, self.valid_loader = self.create_dataloaders()

    def load_data(self): # TODO, finalize this when dataset is merged to main
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        """Assuming data is dictionary output of one NWB file, change later"""
        data = load_one_nwb(self.data_path)
        train_X, valid_X, train_Y, valid_Y = prep_data_and_split(data, self.sequence_length, self.num_train_trials)
        return train_X, train_Y, valid_X, valid_Y
    
    def create_dataloaders(self):
        """Creates PyTorch DataLoaders for training and validation data."""
        train_dataset = TensorDataset(self.train_X.detach().clone().to(torch.float32), 
                                    self.train_Y.detach().clone().to(torch.float32))
        valid_dataset = TensorDataset(self.valid_X.detach().clone().to(torch.float32), 
                                    self.valid_Y.detach().clone().to(torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader

    def train_model(self, train_loader = None, valid_loader = None):
        if(train_loader is not None):
            self.train_loader = train_loader
        if(valid_loader is not None):
            self.valid_loader = valid_loader

        self.logger = {'train_loss': [], 'val_loss': [], 'correlation': []}
        for epoch in range(self.num_epochs):

            # Train for one epoch
            train_loss = self.model._train_one_epoch(self.train_loader, self.model, self.optimizer, self.loss_func, self.device)

            # Validate after each epoch
            val_loss, correlation = self.model._validate_one_epoch(self.train_loader, self.model, self.loss_func, self.device)

            # Record losses and correlation
            self.logger['train_loss'].append(train_loss)
            self.logger['val_loss'].append(val_loss)
            self.logger['correlation'].append(correlation)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Print progress
            if self.print_results and (epoch % self.print_every == 0 or epoch == self.num_epochs - 1):
                print(f"Epoch {epoch}/{self.num_epochs - 1}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Correlation: {correlation[0]:.4f}, {correlation[1]:.4f}")

            if self.print_results:
                print("*** Training Complete ***")

        return self.model, self.logger


