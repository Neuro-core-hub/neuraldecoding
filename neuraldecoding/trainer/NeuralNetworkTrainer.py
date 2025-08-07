import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
import neuraldecoding.utils.loss_functions as loss_functions
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
        self.model = self.create_model(OmegaConf.merge(config.model, config.preprocessing)).to(self.device)
        self.optimizer = self.create_optimizer(config.optimizer, self.model.parameters())
        self.scheduler, self.scheduler_params = self.create_scheduler(config.scheduler, self.optimizer)
        self.loss_func = self.create_loss_function(config.loss_func)
        self.num_epochs = config.training.num_epochs
        self.max_iters = config.training.max_iters
        self.batch_size = config.training.batch_size
        self.clear_cache = config.training.clear_cache
        # Evaluation and logging params
        self.print_results = config.training.get("print_results", True)
        self.print_every = config.training.get("print_every", 10)
        self.metrics = config.evaluation.metrics
        self.metric_params = config.evaluation.get("params", {})
        self.logger = {metric: [[], []] for metric in self.metrics}
        self.preprocessor = preprocessor
        self.data_path = config.data.data_path
        self.train_loader, self.valid_loader, self.test_loader = self.create_dataloaders()

        assert self.scheduler is not None or self.max_iters is not None or self.num_epochs is not None, "At least one of scheduler, max_iters, or num_epochs must be defined."

    def create_dataloaders(self):
        """Creates PyTorch DataLoaders for training and validation data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        """Assuming data is dictionary output of one NWB file, change later"""
        data = load_one_nwb(self.data_path)
        data_tuple = self.preprocessor.preprocess_pipeline(data, params={'is_train': True})

        if len(data_tuple) == 2:
            self.train_ds, self.valid_ds = data_tuple
        elif len(data_tuple) == 3:
            self.train_ds, self.valid_ds, self.test_ds = data_tuple

        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False)

        if hasattr(self, 'test_ds'):
            test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
        else:
            test_loader = None

        return train_loader, valid_loader, test_loader

    def create_optimizer(self, optimizer_config: DictConfig, model_params) -> Optimizer:
        """Creates and returns an optimizer based on the configuration."""
        optimizer_class = getattr(torch.optim, optimizer_config.type)
        return optimizer_class(model_params, **optimizer_config.params)

    def create_scheduler(self, scheduler_config: DictConfig, optimizer: Optimizer) -> _LRScheduler:
        """Creates and returns a learning rate scheduler based on the configuration."""
        if scheduler_config is None:
            return None
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config.type)
        scheduler_params = {}
        scheduler_params['one_cycle'] = scheduler_config.get('one_cycle', False)
        scheduler_params['min_lr'] = scheduler_config.params.get('min_lr', 0.0)

        return scheduler_class(optimizer, **scheduler_config.params), scheduler_params

    def create_loss_function(self, loss_config: DictConfig):
        """Creates and returns a loss function based on the configuration."""
        # Try to get from torch.nn first (for built-in losses)
        loss_class = getattr(torch.nn, loss_config.type, None)
        if loss_class is not None:
            return loss_class(**loss_config.params)

        # Then try to get from your custom loss functions
        loss_func = getattr(loss_functions, loss_config.type, None)
        if loss_func is not None:
            return lambda y_true, y_pred, **kwargs: loss_func(y_true, y_pred, **{**loss_config.params, **kwargs})

        raise ValueError(f"Loss function '{loss_config.type}' not found in torch.nn or neuraldecoding.utils.loss_functions.")
        
    def create_model(self, model_config: DictConfig) -> torch.nn.Module:
        """Creates and returns a loss function based on the configuration."""
        model_class = getattr(neuraldecoding.model.neural_network_models, model_config.type)
        model = model_class(model_config.params)
        return model

    def train_model(self):
        epoch = 1
        iteration = 1
        while True:
            # Train
            self.model.train()
            running_loss = 0.0
            train_all_predictions = []
            train_all_targets = []
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                x = batch['neu'].to(self.device)
                y = batch['kin'].to(self.device)

                loss, yhat = self.model.train_step(x.to(self.device), y.to(self.device), self.model, self.optimizer, self.loss_func, clear_cache = self.clear_cache)

                if(self.clear_cache):
                    del y, yhat

                running_loss += loss.item()

                if iteration != 0 and iteration % self.print_every == 0:
                    train_loss = running_loss / len(self.train_loader)

                    # Validate
                    self.model.eval()
                    running_val_loss = 0.0
                    val_all_predictions = []
                    val_all_targets = []
                    with torch.no_grad():
                        x_val = torch.stack([sample['neu'] for sample in self.valid_loader.dataset]).to(self.device)
                        y_val = torch.stack([sample['kin'] for sample in self.valid_loader.dataset]).to(self.device)
                        yhat_val = self.model(x_val)
                        val_loss = self.loss_func(yhat_val, y_val)

                        running_val_loss += val_loss.item()
                        val_all_predictions = yhat_val.cpu().numpy()
                        val_all_targets = y_val.cpu().numpy()
                        if(self.clear_cache):
                            del y_val, yhat_val

                    val_all_predictions = np.concatenate(val_all_predictions, axis=0)
                    val_all_targets = np.concatenate(val_all_targets, axis=0)
                    val_loss = running_val_loss / len(self.valid_loader)

                    with torch.no_grad():
                        x_train = torch.stack([sample['neu'] for sample in self.train_loader.dataset]).to(self.device)
                        y_train = torch.stack([sample['kin'] for sample in self.train_loader.dataset]).to(self.device)
                        yhat_train = self.model(x_train)
                        train_all_predictions = yhat_train.cpu().numpy()
                        train_all_targets = y_train.cpu().numpy()
                        if(self.clear_cache):
                            del y_train, yhat_train

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
                        if self.scheduler_params['one_cycle']:
                            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.scheduler.step(val_loss)
                                if self.optimizer.param_groups[0]['lr'] < self.scheduler_params['min_lr']:
                                    return self.model, self.logger
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
                
                if self.max_iters is not None and iteration >= self.max_iters:
                    return self.model, self.logger
                iteration += 1
            
            if self.scheduler and not self.scheduler_params['one_cycle']:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            if epoch >= self.num_epochs and self.num_epochs is not None:
                return self.model, self.logger
            
            epoch += 1


    def clear_gpu_cache(self):
        self.model.cpu()
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()