import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from neuraldecoding.utils import prep_data_and_split, load_one_nwb
import neuraldecoding.model.neural_network_models
from neuraldecoding.trainer.Trainer import Trainer
import neuraldecoding.stabilization.latent_space_alignment
from neuraldecoding.model.neural_network_models.NeuralNetworkModel import NeuralNetworkModel
from neuraldecoding.model.neural_network_models.LSTM import LSTM
import neuraldecoding.utils.eval_metrics
from neuraldecoding.utils.eval_metrics import *
import os

class NNTrainer(Trainer):
    def __init__(self, preprocessor, config):
        super().__init__()
        # General training params 
        self.device = torch.device(config.training.device)
        self.model = self.create_model(config.model).to(self.device)
        self.optimizer = self.create_optimizer(config.optimizer, self.model.parameters())
        self.scheduler = self.create_scheduler(config.scheduler, self.optimizer)
        self.loss_func = self.create_loss_function(config.loss_func)
        self.num_epochs = config.training.num_epochs
        self.batch_size = config.training.batch_size
        self.clear_cache = config.training.clear_cache
        # Evaluation and logging params
        self.print_results = config.training.get("print_results", True)
        self.print_every = config.training.get("print_every", 10)
        self.metrics = config.evaluation.metrics
        self.metric_params = config.evaluation.get("params", {})
        self.logger = {metric: [[], []] for metric in self.metrics}
        # Data specific params, TODO: change when dataset is finalized
        self.preprocessor = preprocessor
        self.data_path = config.data.data_path
        self.train_X, self.train_Y, self.valid_X, self.valid_Y = self.load_data()
        self.train_loader, self.valid_loader = self.create_dataloaders()

    def load_data(self): # TODO, finalize this when dataset is merged to main
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        """Assuming data is dictionary output of one NWB file, change later"""
        data = load_one_nwb(self.data_path)
        train_X, valid_X, train_Y, valid_Y = self.preprocessor.preprocess_pipeline(data, params={'is_train': True})
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
        model_class = getattr(neuraldecoding.model.neural_network_models, model_config.type)
        model = model_class(model_config.params)
        return model

    def train_model(self, train_loader = None, valid_loader = None):
        # Override loaders if provided
        if(train_loader is not None):
            self.train_loader = train_loader
        if(valid_loader is not None):
            self.valid_loader = valid_loader

        for epoch in range(self.num_epochs):
            # Train
            self.model.train()
            running_loss = 0.0
            train_all_predictions = []
            train_all_targets = []
            for x,y in self.train_loader:
                x = neuraldecoding.utils.add_training_noise(x, 0.1, 0,2, device='cuda') #bad practice! just for validation for now
                self.optimizer.zero_grad()

                loss, yhat = self.model.train_step(x.to(self.device), y.to(self.device), self.model, self.optimizer, self.loss_func, clear_cache = self.clear_cache)

                running_loss += loss.item()
                train_all_predictions.append(yhat.detach().cpu().numpy())
                train_all_targets.append(y.detach().cpu().numpy())
                if(self.clear_cache):
                    del y, yhat

            train_all_predictions = np.concatenate(train_all_predictions, axis=0)
            train_all_targets = np.concatenate(train_all_targets, axis=0)
            train_loss = running_loss / len(self.train_loader)

            # Validate
            self.model.eval()
            running_val_loss = 0.0
            val_all_predictions = []
            val_all_targets = []
            with torch.no_grad():
                for x_val, y_val in self.valid_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    yhat_val = self.model(x_val)
                    val_loss = self.loss_func(yhat_val, y_val)

                    running_val_loss += val_loss.item()
                    val_all_predictions.append(yhat_val.cpu().numpy())
                    val_all_targets.append(y_val.cpu().numpy())
                    if(self.clear_cache):
                        del y_val, yhat_val

            val_all_predictions = np.concatenate(val_all_predictions, axis=0)
            val_all_targets = np.concatenate(val_all_targets, axis=0)
            val_loss = running_val_loss / len(self.valid_loader)

            # Calculate and populate metrics
            for metric in self.metrics:
                if metric == "loss":
                    self.logger[metric][0].append(train_loss)
                    self.logger[metric][1].append(val_loss)
                else:
                    metric_param = self.metric_params.get(metric, None)
                    metric_class = getattr(neuraldecoding.utils.eval_metrics, metric)
                    self.logger[metric][0].append(metric_class(train_all_predictions, train_all_targets, metric_param))
                    self.logger[metric][1].append(metric_class(val_all_predictions, val_all_targets, metric_param))

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print progress
            if self.print_results and (epoch % self.print_every == 0 or epoch == self.num_epochs - 1):
                print(f"Epoch {epoch}/{self.num_epochs - 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                for metric in self.logger:
                    train_metric = self.logger[metric][0][-1]
                    val_metric = self.logger[metric][1][-1]
                    print(f"    {metric:>12}{': train = ':>12}{train_metric}")
                    print(f"    {'':>12}{'  val = ':>12}{val_metric}")

        return self.model, self.logger


    def clear_gpu_cache(self):
        self.model.cpu()
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()