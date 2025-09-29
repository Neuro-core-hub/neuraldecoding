from omegaconf import DictConfig
import torch
import json
import collections
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from neuraldecoding.utils import loss_functions
from neuraldecoding.trainer.Trainer import Trainer
from neuraldecoding.utils import eval_metrics
from omegaconf import open_dict
from neuraldecoding.model import neural_network_models
import numpy as np
import neuraldecoding
import warnings
import copy

class NNTrainer(Trainer):
    def __init__(self, dataset, preprocessor, config):
        super().__init__(config)
        # General training params
        self.device = torch.device(config.training.device)
        self.model = self.create_model(config.model).to(self.device)
        self.optimizer = self.create_optimizer(config.optimizer, self.model.parameters())
        self.scheduler, self.scheduler_params = self.create_scheduler(config.scheduler, self.optimizer)
        self.loss_func = self.create_loss_function(config.loss_func)
        self.num_epochs = config.training.num_epochs
        self.max_iters = config.training.max_iters
        self.take_best = config.training.get('take_best', True)
        if config.model.type == 'LSTMTrialInput':
            self.batch_size = config.training.get('batch_size', 1)
            if self.batch_size != 1:
                warnings.warn("LSTMTrialInput does not support batch_size in model config. Setting batch_size to 1.")
                with open_dict(config.training):
                    self.batch_size = 1  # Override batch size to 1 for trial input support
        else:
            self.batch_size = config.training.get('batch_size', 64)
        self.clear_cache = config.training.clear_cache
        # Evaluation and logging params
        self.print_results = config.training.get("print_results", True)
        self.print_every = config.training.get("print_every", 10)
        self.metrics = config.evaluation.metrics
        self.metric_params = config.evaluation.get("params", {})
        self.only_val = config.evaluation.get("only_val", [False]*len(self.metrics))
        self.logger = {metric: [[], []] for metric in self.metrics}
        self.preprocessor = preprocessor
        self.train_ds, self.valid_ds, self.test_ds = None, None, None
        self.saved_data = None
        self.train_loader, self.valid_loader, self.test_loader = self.create_dataloaders(dataset)

        assert self.scheduler is not None or self.max_iters is not None or self.num_epochs is not None, "At least one of scheduler, max_iters, or num_epochs must be defined."

    def create_dataloaders(self, dataset):
        """Creates PyTorch DataLoaders for training and validation data."""
        data_tuple, self.saved_data = self.preprocessor.preprocess_pipeline(dataset, params={'is_train': True})
        print("Saved data keys from preprocessing: ", self.saved_data.keys())

        if len(data_tuple) == 2:
            self.train_ds, self.valid_ds = data_tuple
        elif len(data_tuple) == 3:
            self.train_ds, self.valid_ds, self.test_ds = data_tuple

        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(self.valid_ds, batch_size=len(self.valid_ds), shuffle=False)

        if self.test_ds is not None:
            test_loader = DataLoader(self.test_ds, batch_size=len(self.test_ds), shuffle=False)
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
        scheduler_params = {}
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config.type)
        if scheduler_config.type == 'ReduceLROnPlateau':
            learning_rate = optimizer.param_groups[0]['lr']
            scheduler_params['min_lr'] = learning_rate / (scheduler_config.get('num_steps', 2) + 0.001)
        scheduler_params['n_cycle'] = scheduler_config.get('n_cycle', False)

        return scheduler_class(optimizer, **scheduler_config.params), scheduler_params

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

    def train_model(self):
        epoch = 1
        iteration = 1
        best_val_loss = float('inf')
        best_state_dict = None
        epoch_best = 0
        iteration_best = 0

        while True:
            # Train
            running_loss = 0.0
            i = 0
            for batch in self.train_loader:
                i += 1
                self.model.train()
                self.optimizer.zero_grad()

                loss, yhat = self.model.train_step(batch, self.model, self.optimizer, self.loss_func, clear_cache = self.clear_cache)

                if(self.clear_cache):
                    del y, yhat

                running_loss += loss.item()

                if iteration != 0 and iteration % self.print_every == 0:
                    train_loss = running_loss / i

                    # Validate
                    self.model.eval()
                    running_val_loss = 0.0
                    with torch.no_grad():
                        x_val = torch.stack([sample['neu'] for sample in self.valid_loader.dataset]).to(self.device)
                        y_val = torch.stack([sample['kin'] for sample in self.valid_loader.dataset]).to(self.device)
                        yhat_val = self.model(x_val)
                        val_loss = self.loss_func(yhat_val, y_val)
                        running_val_loss += val_loss.item()

                        if self.take_best and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_state_dict = copy.deepcopy(self.model.state_dict())
                            epoch_best = epoch
                            iteration_best = iteration

                        if(self.clear_cache):
                            del y_val, yhat_val

                    with torch.no_grad():
                        x_train = torch.stack([sample['neu'] for sample in self.train_loader.dataset]).to(self.device)
                        y_train = torch.stack([sample['kin'] for sample in self.train_loader.dataset]).to(self.device)
                        yhat_train = self.model(x_train)
                        if(self.clear_cache):
                            del y_train, yhat_train

                    # Calculate and populate metrics
                    for metric in self.metrics:
                        if metric == "loss":
                            if not self.only_val[self.metrics.index(metric)]:
                                self.logger[metric][0].append(train_loss)
                            else:
                                self.logger[metric][0].append('not computed')
                            self.logger[metric][1].append(val_loss)
                        else:
                            metric_param = self.metric_params.get(metric, None)
                            metric_class = getattr(eval_metrics, metric)
                            if not self.only_val[self.metrics.index(metric)]:
                                self.logger[metric][0].append(metric_class(yhat_train, y_train, metric_param))
                            else:
                                self.logger[metric][0].append('not computed')
                            self.logger[metric][1].append(metric_class(yhat_val, y_val, metric_param))

                    # Scheduler step
                    if self.scheduler:
                        if self.scheduler_params['n_cycle']:
                            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.scheduler.step(val_loss)
                                if self.optimizer.param_groups[0]['lr'] < self.scheduler_params['min_lr']:
                                    if self.take_best and best_state_dict is not None:
                                        self.reload_model(best_state_dict, epoch_best, iteration_best)
                                    return self.model, self.logger
                            else:
                                self.scheduler.step()

                    # Print progress
                    if self.print_results:
                        print(f"Epoch {epoch}, iteration {iteration}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                        for metric in self.logger:
                            train_metric = self.logger[metric][0][-1]
                            val_metric = self.logger[metric][1][-1]
                            print(f"    {metric:>12}{': train = ':>12}{train_metric}")
                            print(f"    {'':>12}{'  val = ':>12}{val_metric}")
                
                if self.max_iters is not None and iteration >= self.max_iters:
                    if self.take_best and best_state_dict is not None:
                        self.reload_model(best_state_dict, epoch_best, iteration_best)
                    return self.model, self.logger
                iteration += 1

                self.model.train()
            
            if self.scheduler and not self.scheduler_params['n_cycle']:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                    if self.optimizer.param_groups[0]['lr'] < self.scheduler_params['min_lr']:
                        if self.take_best and best_state_dict is not None:
                            self.reload_model(best_state_dict, epoch_best, iteration_best)
                        return self.model, self.logger
                else:
                    self.scheduler.step()

            if self.num_epochs is not None and epoch >= self.num_epochs:
                if self.take_best and best_state_dict is not None:
                    self.reload_model(best_state_dict, epoch_best, iteration_best)
                return self.model, self.logger
            
            epoch += 1


    def clear_gpu_cache(self):
        self.model.cpu()
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()
    
    def reload_model(self, model_state_dict, epoch_best, iteration_best):
        """Reloads the model with the given state dictionary."""
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"Reloaded model from epoch {epoch_best}, iteration {iteration_best} with best validation loss.")

        with torch.no_grad():
            x_val = torch.stack([sample['neu'] for sample in self.valid_loader.dataset]).to(self.device)
            y_val = torch.stack([sample['kin'] for sample in self.valid_loader.dataset]).to(self.device)
            yhat_val = self.model(x_val)
            val_loss = self.loss_func(yhat_val, y_val)

            if(self.clear_cache):
                del x_val, y_val, yhat_val
                del x_train, y_train, yhat_train

        final_metrics = {}
        for metric in self.metrics:
            if metric == "loss":
                final_metrics[metric] = val_loss
            else:
                metric_param = self.metric_params.get(metric, None)
                metric_class = getattr(eval_metrics, metric)
                final_metrics[metric]  = metric_class(yhat_val, y_val, metric_param)

        print(f"Final validation metrics: ")
        for metric in self.logger:
            val_metric = final_metrics[metric]
            print(f"    {metric:>12}{'  val = ':>12}{val_metric}")

        return self.model

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