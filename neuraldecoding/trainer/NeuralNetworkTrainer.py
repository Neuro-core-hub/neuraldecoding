import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import collections
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from neuraldecoding.utils import loss_functions
from neuraldecoding.trainer.Trainer import Trainer
from neuraldecoding.model import neural_network_models
import neuraldecoding
import warnings

class NNTrainer(Trainer):
    def __init__(self, preprocessor, config, dataset = None):
        super().__init__(config)
        # General training params
        self.device = torch.device(config.training.device)
        self.model = self.create_model(config.model).to(self.device)
        self.optimizer = self.create_optimizer(config.optimizer, self.model.parameters())
        self.scheduler = self.create_scheduler(config.scheduler, self.optimizer)
        self.loss_func = self.create_loss_function(config.loss_func)
        self.num_epochs = config.training.get('num_epochs', None)
        self.batch_size = config.training.get('batch_size', 64)
        self.full_batch_valid = config.training.get('full_batch_valid', False)
        if config.model.type == 'LSTMTrialInput':
            if self.batch_size != 1:
                warnings.warn("LSTMTrialInput does not support batch_size in model config. Setting batch_size to 1.")
                self.batch_size = 1  # Override batch size to 1 for trial input support
        if isinstance(self.batch_size, collections.abc.Iterable):
            self.train_batch_size = self.batch_size[0]
            self.valid_batch_size = self.batch_size[1]
        else:
            self.train_batch_size = self.batch_size
            self.valid_batch_size = self.batch_size
        self.clear_cache = config.training.clear_cache
        self.preprocessor = preprocessor
        if dataset is not None:
            self.data_dict = self.load_data(dataset)
            self.train_loader, self.valid_loader = self.create_dataloaders()
                
    def load_data(self, data): # TODO, finalize this when dataset is merged to main
        data_dict = self.preprocessor.preprocess_pipeline(data, params={'is_train': True})
        # you should have Dict2TrainerBlock in your pipeline
        # from block Dict2TrainerBlock, outputing A dictionary containing either:
		#                               - 'X' and 'Y' if 2 keys are present
		#                               - 'X_train', 'X_val', 'Y_train', 'Y_val' if 4 keys are present
        # Typically it is 4 keys for NN trainer, so it will be 'X_train', 'X_val', 'Y_train', 'Y_val'.
        return data_dict
    
    def create_dataloaders(self):
        """Creates PyTorch DataLoaders for training and validation data."""
        train_dataset = TensorDataset(self.data_dict["X_train"].detach().clone().to(torch.float32), 
                                    self.data_dict["Y_train"].detach().clone().to(torch.float32))
        valid_dataset = TensorDataset(self.data_dict['X_val'].detach().clone().to(torch.float32), 
                                    self.data_dict['Y_val'].detach().clone().to(torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        if self.full_batch_valid:
            valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
        else:
            valid_loader = DataLoader(valid_dataset, batch_size=self.valid_batch_size, shuffle=False)
        return train_loader, valid_loader

    def create_optimizer(self, optimizer_config: DictConfig, model_params) -> Optimizer:
        """Creates and returns an optimizer based on the configuration."""
        optimizer_class = getattr(torch.optim, optimizer_config.type)
        return optimizer_class(model_params, **optimizer_config.params)
    
    def create_scheduler(self, scheduler_config: DictConfig, optimizer: Optimizer) -> _LRScheduler:
        """Creates and returns a learning rate scheduler based on the configuration."""
        if scheduler_config is None or not scheduler_config or len(scheduler_config) == 0:
            return None
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config.type)
        return scheduler_class(optimizer, **scheduler_config.params)

    def create_loss_function(self, loss_config: DictConfig):
        """Creates and returns a loss function based on the configuration."""
        # Try to get from torch.nn first (for built-in losses)
        loss_class = getattr(torch.nn, loss_config.type, None)
        if loss_class is not None:
            return loss_class(**loss_config.params)

        # Then try to get from custom loss functions
        loss_class = getattr(loss_functions, loss_config.type, None)
        if loss_class is not None:
            return loss_class(**loss_config.params)

        raise ValueError(f"Loss function '{loss_config.type}' not found in torch.nn or neuraldecoding.utils.loss_functions.")
        
    def create_model(self, model_config: DictConfig) -> torch.nn.Module:
        """Creates and returns a model based on the configuration."""
        model_class = getattr(neural_network_models, model_config.type)
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
                self.optimizer.zero_grad()

                loss, yhat, y = self.model.train_step(x.to(self.device), y.to(self.device), self.optimizer, self.loss_func, clear_cache = self.clear_cache, return_y=True)

                running_loss += loss.item()
                train_all_predictions.append(yhat.detach().cpu().numpy())
                train_all_targets.append(y.detach().cpu().numpy())
                if(self.clear_cache):
                    del y, yhat

            train_loss = running_loss / len(self.train_loader)

            # Validate
            val_loss, val_all_predictions, val_all_targets = self.validate_model()

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Calculate and populate metrics
            for metric in self.metrics:
                if metric == "loss":
                    self.logger[metric]['train'].append(train_loss)
                    self.logger[metric]['valid'].append(val_loss)
                else:
                    metric_param = self.metric_params.get(metric, None)
                    metric_class = getattr(neuraldecoding.utils.eval_metrics, metric)
                    self.logger[metric]['valid'].append(metric_class(val_all_predictions, val_all_targets, metric_param))
                    train_metrics = []
                    for train_prediction, train_target in zip(train_all_predictions, train_all_targets):
                        train_metrics.append(metric_class(train_prediction, train_target, metric_param))
                    self.logger[metric]['train'].append(train_metrics)

            # Logging
            self.save_print_log(epoch, train_loss, val_loss)

        return self.model, self.logger

    def clear_gpu_cache(self):
        self.model.cpu()
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()

    def validate_model(self):
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

        return val_loss, val_all_predictions, val_all_targets

class LSTMTrainer(NNTrainer):
    def __init__(self, preprocessor, config, dataset = None):
        super().__init__(preprocessor, config, dataset)

    def validate_model(self):
        # Validate
        self.model.eval()
        running_val_loss = 0.0
        val_all_predictions = []
        val_all_targets = []
        h = None

        with torch.no_grad():
            all_x = self.valid_loader.dataset.tensors[0][:,:,-1]
            val_all_targets = self.valid_loader.dataset.tensors[1]
            val_all_predictions = self.model.forward(all_x, return_all_tsteps=True)
            val_loss = self.loss_func(val_all_predictions, val_all_targets).item()

        return val_loss, val_all_predictions.detach().cpu().numpy(), val_all_targets.detach().cpu().numpy()

class IterationNNTrainer(NNTrainer):
    '''
    The trainer used in LINK dataset multiday training. Archived here for reference.
    Does NOT do logging.

    Based on Joey's training code for LINK dataset BCI-decoding section.
    '''
    def __init__(self, preprocessor, config, dataset = None):
        super().__init__(preprocessor, config, dataset = None)
    
    def train_model(self, train_loader=None, valid_loader=None):
        # Override loaders if provided
        if(train_loader is not None):
            self.train_loader = train_loader
        if(valid_loader is not None):
            self.valid_loader = valid_loader
        iteration = 0

        while iteration < self.num_epochs:
            for x,y in self.train_loader:
                self.model.train()
                if iteration >= self.num_epochs:
                    break

                self.optimizer.zero_grad()

                loss, yhat = self.model.train_step(x.to(self.device), y.to(self.device), self.optimizer, self.loss_func, clear_cache = self.clear_cache)

                if(self.clear_cache):
                    del y, yhat

                # Validate
                self.model.eval()
                total_loss = 0.0
                num_batches = 0
                with torch.no_grad():
                    for x_val, y_val in self.valid_loader:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)
                        yhat_val = self.model(x_val)
                        val_loss = self.loss_func(yhat_val, y_val)
                        total_loss += val_loss.item()
                        num_batches += 1
                if self.print_results and (iteration % self.print_every == 0 or iteration == self.num_epochs - 1):
                    print(f"Iteration {iteration}, Train Loss: {loss.item():.4f}, Val Loss: {(total_loss / num_batches):.4f}")

                # Scheduler step
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                iteration += 1
        return self.model, self.logger

class TCFNNTrainer(NNTrainer):
    '''
    Trainer for the tcFNN model.
    '''
    def __init__(self, preprocessor, config, dataset = None):
        super().__init__(preprocessor, config, dataset = dataset)
    

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
                self.optimizer.zero_grad()

                loss, yhat = self.model.train_step(x.to(self.device), y.to(self.device), self.optimizer, self.loss_func, clear_cache = self.clear_cache)

                running_loss += loss.item()
                train_all_predictions.append(yhat.detach().cpu().numpy())
                train_all_targets.append(y.detach().cpu().numpy())
                if(self.clear_cache):
                    del y, yhat

            train_all_predictions = np.concatenate(train_all_predictions, axis=0)
            train_all_targets = np.concatenate(train_all_targets, axis=0)
            train_loss = running_loss / len(self.train_loader)

            # Validate
            val_loss, val_all_predictions, val_all_targets = self.validate_model()

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Calculate and populate metrics
            for metric in self.metrics:
                if metric == "loss":
                    self.logger[metric]['train'].append(train_loss)
                    self.logger[metric]['valid'].append(val_loss)
                else:
                    metric_param = self.metric_params.get(metric, None)
                    metric_class = getattr(neuraldecoding.utils.eval_metrics, metric)
                    self.logger[metric]['train'].append(metric_class(train_all_predictions, train_all_targets, metric_param))
                    self.logger[metric]['valid'].append(metric_class(val_all_predictions, val_all_targets, metric_param))
            
            # Logging
            self.save_print_log(epoch, train_loss, val_loss)

        self.model.scaler.fit(self.model, self.train_loader, device=self.device, dtype=torch.float32, num_outputs=self.model.num_states, verbose=False)
        return self.model, self.logger

    def validate_model(self):
        # Validate
        self.model.eval()
        running_val_loss = 0.0
        val_all_predictions = []
        val_all_targets = []
        
        self.model.scaler.fit(self.model, self.train_loader, device=self.device, dtype=torch.float32, num_outputs=self.model.num_states, verbose=False)

        with torch.no_grad():
            for x_val, y_val in self.valid_loader:
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                yhat_val = self.model.scaler.unscale(self.model(x_val))
                val_loss = self.loss_func(yhat_val, y_val)

                running_val_loss += val_loss.item()
                val_all_predictions.append(yhat_val.cpu().numpy())
                val_all_targets.append(y_val.cpu().numpy())
                if(self.clear_cache):
                    del y_val, yhat_val

        val_all_predictions = np.concatenate(val_all_predictions, axis=0)
        val_all_targets = np.concatenate(val_all_targets, axis=0)
        val_loss = running_val_loss / len(self.valid_loader)

        return val_loss, val_all_predictions, val_all_targets