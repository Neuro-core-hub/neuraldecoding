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
from neuraldecoding.utils.special_datasets import BehaviorDatasetCustom
import neuraldecoding
import warnings
import copy
import sklearn.preprocessing

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
        self.max_iters = config.training.get('max_iters', None)
        self.print_on = config.training.get('print_on', 'epoch')
        self.min_lr_plateau = config.training.get('min_lr_plateau', 0)
        self.take_best = config.training.get('take_best', False)
        self.batch_size = config.training.get('batch_size', 64)
        self.full_batch_train = config.training.get('full_batch_train', False)
        self.full_batch_valid = config.training.get('full_batch_valid', False)
        self.trialized_loss = config.training.get('trialized_loss', False)
        if config.loss_func.type == 'LSTMTrialInput':
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
        if 'behavior_train_normalizer' in preprocessor.saved_data:
            self.model.behavior_scaler = preprocessor.saved_data['behavior_train_normalizer']
        elif config.model.type == 'LSTMTrialInput_RankDist':
            pass # Rank loss behavior scaler handled in child class, calculated after training
        else:
            warnings.warn("No behavior scaler found in preprocessor saved data.")
            self.model.behavior_scaler = None
        if 'neural_train_normalizer' in preprocessor.saved_data:
            self.model.neural_scaler = preprocessor.saved_data['neural_train_normalizer']
        else:
            warnings.warn("No neural scaler found in preprocessor saved data.")
            self.model.neural_scaler = None
                
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
        if self.full_batch_train:
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True, drop_last=True)
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

        iteration = 0
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        best_iteration = 0
            
        for epoch in range(self.num_epochs):
            # Train
            running_loss = 0.0
            within_epoch_iteration = 0
            train_all_predictions = []
            train_all_targets = []

            for x,y in self.train_loader:
                self.model.train()
                self.optimizer.zero_grad()
                if self.trialized_loss:
                    if iteration == 0:
                        n_bins = self.preprocessor.saved_data['neural_train_denorm_data'].shape[0]
                        train_ends_mask = self.preprocessor.saved_data['bin_trial_end_idx'] < n_bins
                        starts = self.preprocessor.saved_data['bin_trial_start_idx'][train_ends_mask]
                        ends = self.preprocessor.saved_data['bin_trial_end_idx'][train_ends_mask]

                    loss, yhat, y = self.model.train_step(x.to(self.device), y.to(self.device), starts, ends, self.optimizer, self.loss_func, clear_cache = self.clear_cache, return_y=True)
                else:
                    loss, yhat, y = self.model.train_step(x.to(self.device), y.to(self.device), self.optimizer, self.loss_func, clear_cache = self.clear_cache, return_y=True)

                running_loss += loss.item()
                train_all_predictions.append(yhat.detach().cpu().numpy())
                train_all_targets.append(y.detach().cpu().numpy())
                if(self.clear_cache):
                    del y, yhat
                
                iteration += 1
                within_epoch_iteration += 1
                
                if self.print_on == 'iters':
                    
                    if iteration % self.print_every == 0:
                        train_loss = running_loss / within_epoch_iteration
                        
                        val_loss, val_all_predictions, val_all_targets = self.validate_model()

                        # Save best model
                        if self.take_best and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_state = copy.deepcopy(self.model.state_dict())
                            best_epoch = epoch
                            best_iteration = iteration
                        self.update_scheduler(val_loss)
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            if self.optimizer.param_groups[0]['lr'] <= self.min_lr_plateau:
                                print(f"Learning rate has reached minimum threshold ({self.min_lr_plateau}). Ending training.")
                                if self.take_best and best_model_state is not None:
                                    self.model.load_state_dict(best_model_state)
                                    print(f"Loaded best model from epoch {best_epoch}, iteration {best_iteration} with validation loss: {best_val_loss:.4f}")
                                return self.model, self.logger
                    
                        self.update_logger(train_loss, val_loss, train_all_predictions, train_all_targets, val_all_predictions, val_all_targets, epoch, iteration)

                if self.max_iters is not None and iteration >= self.max_iters:
                    print(f"Reached maximum iterations ({self.max_iters}). Ending training.")
                    if self.take_best and best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                        print(f"Loaded best model from epoch {best_epoch}, iteration {best_iteration} with validation loss: {best_val_loss:.4f}")
                    return self.model, self.logger

            # Update logger
            if self.print_on == 'epoch':
                train_loss = running_loss / len(self.train_loader)
                
                if epoch % self.print_every == 0:
                    # Validate
                    val_loss, val_all_predictions, val_all_targets = self.validate_model()
                    
                    # Save best model
                    if self.take_best and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch
                        best_iteration = iteration

                    self.update_scheduler(val_loss)

                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if self.optimizer.param_groups[0]['lr'] <= self.min_lr_plateau:
                            print(f"Learning rate has reached minimum threshold ({self.min_lr_plateau}). Ending training.")
                            if self.take_best and best_model_state is not None:
                                self.model.load_state_dict(best_model_state)
                                print(f"Loaded best model from epoch {best_epoch}, iteration {best_iteration} with validation loss: {best_val_loss:.4f}")
                            return self.model, self.logger
                    self.update_logger(train_loss, val_loss, train_all_predictions, train_all_targets, val_all_predictions, val_all_targets, epoch, iteration)
        
        print("Reached maximum number of epochs. Ending training.")
        if self.take_best and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model from epoch {best_epoch}, iteration {best_iteration} with validation loss: {best_val_loss:.4f}")
        return self.model, self.logger
    
    def update_scheduler(self, val_loss):
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
    
    def update_logger(self, train_loss, val_loss, train_all_predictions, train_all_targets, val_all_predictions, val_all_targets, epoch, iteration):
        for metric, compute_train in zip(self.metrics, self.compute_train):
            # Calculate and populate metrics
            if metric == "loss":
                self.logger[metric]['train'].append(train_loss)
                self.logger[metric]['valid'].append(val_loss)
            else:
                if compute_train:
                    metric_param = self.metric_params.get(metric, None)
                    metric_class = getattr(neuraldecoding.utils.eval_metrics, metric)
                    self.logger[metric]['valid'].append(metric_class(val_all_predictions, val_all_targets, metric_param))
                    train_metrics = []
                    for train_prediction, train_target in zip(train_all_predictions, train_all_targets):
                        train_metrics.append(metric_class(train_prediction, train_target, metric_param))
                    self.logger[metric]['train'].append(train_metrics)
                else:
                    metric_param = self.metric_params.get(metric, None)
                    metric_class = getattr(neuraldecoding.utils.eval_metrics, metric)
                    self.logger[metric]['valid'].append(metric_class(val_all_predictions, val_all_targets, metric_param))
                    self.logger[metric]['train'].append(None)

        # logging
        self.save_print_log_v2(epoch, iteration, train_loss, val_loss)

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

class NNTrialInputTrainer(NNTrainer):
    def __init__(self, preprocessor, config, dataset = None):        
        super().__init__(preprocessor, config, dataset)

    def create_dataloaders(self):
        """Creates PyTorch DataLoaders for trial inputs of training and validation data."""
        
        self.train_start_indices = self.preprocessor.saved_data['train_bin_trial_start_idx']
        self.train_stop_indices = self.preprocessor.saved_data['train_bin_trial_end_idx']

        train_trial_dataset = [] 

        for start, stop in zip(self.train_start_indices, self.train_stop_indices):
            start, stop = int(start), int(stop)
            train_trial_dataset.append([self.data_dict['X_train'][start:stop], self.data_dict['Y_train'][start:stop]])
        
        valid_trial_dataset = TensorDataset(self.data_dict['X_val'].detach().clone().to(torch.float32), 
                                    self.data_dict['Y_val'].detach().clone().to(torch.float32))

        train_loader = DataLoader(train_trial_dataset, batch_size=1, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_trial_dataset, batch_size=self.valid_batch_size, shuffle=False)
        return train_loader, valid_loader

class LSTMTrainer(NNTrainer):
    def __init__(self, preprocessor, config, dataset = None):
        super().__init__(preprocessor, config, dataset)
        self.validate_2d = config.training.get('validate_2d', False)

    def validate_model(self):
        # Validate
        self.model.eval()
        running_val_loss = 0.0
        h = None

        with torch.no_grad():
            if self.validate_2d:
                x_val = self.valid_loader.dataset.tensors[0].to(self.device)
                y_val = self.valid_loader.dataset.tensors[1].to(self.device)
                yhat_val = self.model(x_val, h, return_all_tsteps=True)
                val_loss = self.loss_func(yhat_val, y_val)

                running_val_loss += val_loss.item()
                if(self.clear_cache):
                    del y_val, yhat_val
            else:
                for x_val, y_val in self.valid_loader: # Typically only one batch
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    yhat_val = self.model(x_val, h, return_all_tsteps=True)[:, -1, :]
                    val_loss = self.loss_func(yhat_val, y_val)

                    running_val_loss += val_loss.item()
                    if(self.clear_cache):
                        del y_val, yhat_val

        return val_loss, yhat_val.detach().cpu().numpy(), y_val.detach().cpu().numpy()
    
class LSTMRankTrainer(LSTMTrainer):
    def __init__(self, preprocessor, config, dataset = None):
        super().__init__(preprocessor, config, dataset)
        self.fullhist = config.training.get('fullhist', False)
        self.dof_selections = config.training.get('dof_selections', None) # for mixed MSE, Rank scaling
    
    def create_dataloaders(self):
        """Creates PyTorch DataLoaders for training and validation data."""
        train_dataset = BehaviorDatasetCustom(self.data_dict, xkey='neural_train', ykey='behavior_train', otherdatakeys=['directions', 'onsets'], otherdatakeys_data=['directions_train', 'onsets_train'], device=self.device)
        valid_dataset = BehaviorDatasetCustom(self.data_dict, xkey='neural_val', ykey='behavior_val', otherdatakeys=['directions', 'onsets'], otherdatakeys_data=['directions_val', 'onsets_val'], device=self.device)

        train_dataset.onsets = train_dataset.onsets.detach().clone().to(torch.int16).to(self.device)
        valid_dataset.onsets = valid_dataset.onsets.detach().clone().to(torch.int16).to(self.device)
        train_dataset.directions = train_dataset.directions.detach().clone().to(torch.int16).to(self.device)
        valid_dataset.directions = valid_dataset.directions.detach().clone().to(torch.int16).to(self.device)
        if self.full_batch_train:
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)

        if self.full_batch_valid:
            valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
        else:
            valid_loader = DataLoader(valid_dataset, batch_size=self.valid_batch_size, shuffle=False)
        return train_loader, valid_loader

    def train_model(self, train_loader = None, valid_loader = None):
        # Override loaders if provided
        if(train_loader is not None):
            self.train_loader = train_loader
        if(valid_loader is not None):
            self.valid_loader = valid_loader

        iteration = 0
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        best_iteration = 0
        running_loss = 0.0
            
        for epoch in range(self.num_epochs):
            # Train
            train_all_predictions = []
            train_all_targets = []

            for trial in self.train_loader:
                x = trial['neu']
                y = trial['kin']
                directions = trial['directions']
                onsets = trial['onsets']
                
                self.model.train()
                self.optimizer.zero_grad()
                loss, yhat = self.model.train_step(x.to(self.device), y.to(self.device), directions, onsets, self.optimizer, self.loss_func, clear_cache=self.clear_cache)

                running_loss += loss.item()
                train_all_predictions.append(yhat.detach().cpu().numpy())
                train_all_targets.append(y.detach().cpu().numpy())
                if(self.clear_cache):
                    del y, yhat
                
                iteration += 1
                
                if self.print_on == 'iters':
                    
                    if iteration % self.print_every == 0:
                        train_loss = running_loss / self.print_every
                        running_loss = 0.0
                        
                        val_loss, val_all_predictions, val_all_targets = self.validate_model()

                        # Save best model
                        if self.take_best and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_state = self.model.state_dict().copy()
                            best_epoch = epoch
                            best_iteration = iteration
                        self.update_scheduler(val_loss)
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            if self.optimizer.param_groups[0]['lr'] <= self.min_lr_plateau:
                                print(f"Learning rate has reached minimum threshold ({self.min_lr_plateau}). Ending training.")
                                if self.take_best and best_model_state is not None:
                                    self.model.load_state_dict(best_model_state)
                                    print(f"Loaded best model from epoch {best_epoch}, iteration {best_iteration} with validation loss: {best_val_loss:.4f}")
                                self.compute_behavior_scaler()
                                return self.model, self.logger
                    
                        self.update_logger(train_loss, val_loss, train_all_predictions, train_all_targets, val_all_predictions, val_all_targets, epoch, iteration)

                if self.max_iters is not None and iteration >= self.max_iters:
                    print(f"Reached maximum iterations ({self.max_iters}). Ending training.")
                    if self.take_best and best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                        print(f"Loaded best model from epoch {best_epoch}, iteration {best_iteration} with validation loss: {best_val_loss:.4f}")
                    self.compute_behavior_scaler()
                    return self.model, self.logger

            # Update logger
            if self.print_on == 'epoch':
                train_loss = running_loss / len(self.train_loader)
                running_loss = 0.0
                
                if epoch % self.print_every == 0:
                    # Validate
                    val_loss, val_all_predictions, val_all_targets = self.validate_model()
                    
                    # Save best model
                    if self.take_best and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = self.model.state_dict().copy()
                        best_epoch = epoch
                        best_iteration = iteration

                    self.update_scheduler(val_loss)

                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if self.optimizer.param_groups[0]['lr'] <= self.min_lr_plateau:
                            print(f"Learning rate has reached minimum threshold ({self.min_lr_plateau}). Ending training.")
                            if self.take_best and best_model_state is not None:
                                self.model.load_state_dict(best_model_state)
                                print(f"Loaded best model from epoch {best_epoch}, iteration {best_iteration} with validation loss: {best_val_loss:.4f}")
                            self.compute_behavior_scaler()
                            return self.model, self.logger
                    self.update_logger(train_loss, val_loss, train_all_predictions, train_all_targets, val_all_predictions, val_all_targets, epoch, iteration)
        
        print("Reached maximum number of epochs. Ending training.")
        if self.take_best and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model from epoch {best_epoch}, iteration {best_iteration} with validation loss: {best_val_loss:.4f}")
        self.compute_behavior_scaler()
        return self.model, self.logger
    
    def validate_model(self):
        # Validate
        self.model.eval()
        running_val_loss = 0.0
        val_all_predictions = []
        val_all_targets = []
        h = None

        with torch.no_grad():
            val_loss = 0.0
            val_all_predictions = []
            val_all_targets = []
            for trial in self.valid_loader:
                x = trial['neu']
                y = trial['kin']
                directions = trial['directions']
                onsets = trial['onsets']
                if self.fullhist:
                    yhat = self.model.forward(x.to(self.device), return_all_tsteps=True)

                    val_bin_start = self.preprocessor.saved_data['neural_train_denorm_data'].shape[0]
                    onsets = torch.where(onsets != 0, onsets - val_bin_start, onsets)

                    cum_loss = self.loss_func(yhat, y, directions, onsets, fullhist=True)

                    val_all_predictions = yhat.cpu().numpy()
                    val_all_targets = y.cpu().numpy()
                else:
                    trial_length = (~torch.isnan(x[0, 0, :])).sum().max().item() - self.model.leadup
                    # Edge case: if trial didn't fill leadup, we need to remove the leadup before doing forward pass
                    if torch.isnan(x[0, 0, 0]):
                        x = x[:, :, self.model.leadup:]
                        yhat = self.model.forward(x[:, :, :trial_length], return_all_tsteps=True, remove_leadup=False)
                    else:
                        yhat = self.model.forward(x[:, :, :self.model.leadup + trial_length], return_all_tsteps=True, remove_leadup=True)
                    yhat = yhat.permute(0, 2, 1)
                    
                    cum_loss = torch.tensor(0.0, device=yhat.device)
                    for i in range(0, self.model.past + self.model.future):
                        idx = np.arange(i*self.model.n_dofs, (i+1)*self.model.n_dofs)
                        subset = yhat[:, idx, :]
                        onsets_cur = onsets + self.model.past - i  # shift onsets according to how far in the future we're looking
                        loss = self.loss_func(subset, directions, onsets_cur)
                        if loss is None:
                            continue
                        cum_loss += loss
                    
                    yhat_np = np.squeeze(yhat.cpu().numpy().T)
                    y_np = np.squeeze(y.cpu().numpy().T)
                    val_all_predictions.append(yhat_np)
                    val_all_targets.append(y_np[:trial_length, :])
                    
                val_loss += cum_loss.item()
                
            if isinstance(val_all_predictions, list):
                val_all_predictions = np.concatenate(val_all_predictions, axis=0)
                val_all_targets = np.concatenate(val_all_targets, axis=0)

        return val_loss, val_all_predictions, val_all_targets
    
    def compute_behavior_scaler(self):
        if self.dof_selections is not None:
            # Mixed DOF scaler approach
            self.model.eval()
            with torch.no_grad():
                x_full_train = self.data_dict['neural_train'].to(self.device)
                train_predictions = self.model.forward(x_full_train, return_all_tsteps=True).detach().cpu().numpy()
            prompts = self.preprocessor.saved_data['behavior_train_denorm_data']
    
            scalers = []
            for i, selection in enumerate(self.dof_selections):
                if selection == 1: # rank scaler
                    scalers.append(BehaviorScalerRank(PercentileScaler(np.percentile(train_predictions[:, i], 10, axis=0), np.percentile(train_predictions[:, i], 90, axis=0))))
                elif selection == 0: # standard scaler
                    prompts_dof = prompts[:, i]
                    scalers.append(sklearn.preprocessing.StandardScaler().fit(prompts_dof.reshape(-1, 1)))

            behavior_scaler = MixedDOFScaler(scalers)

        else:
            # Traditional approach, scale to 10th-90th percentile
            # After training, compute behavior scaler via 10th-90th percentile scaling on output of model with input x_full_train
            self.model.eval()
            with torch.no_grad():
                x_full_train = self.data_dict['neural_train'].to(self.device)
                train_predictions = self.model.forward(x_full_train, return_all_tsteps=True).detach().cpu().numpy()

            idx = np.arange(self.model.past * self.model.n_dofs, (self.model.past + 1) * self.model.n_dofs)
            train_predictions = train_predictions[:, idx]

            p10 = np.percentile(train_predictions, 10, axis=0)
            p90 = np.percentile(train_predictions, 90, axis=0)
            behavior_scaler_internal = PercentileScaler(p10, p90)

            # We need the inverse_transform method to go from percentile-scaled to 0-1, and not the transform method
            # Creating this dummy class to swap the methods
            behavior_scaler = BehaviorScalerRank(behavior_scaler_internal)

            self.model.behavior_scaler = behavior_scaler

class MixedDOFScaler:
    def __init__(self, scalers):
        self.scalers = scalers

    def transform(self, data):
        return np.column_stack([scaler.transform(data[:, i]) for i, scaler in enumerate(self.scalers)])

    def inverse_transform(self, data):
        return np.column_stack([scaler.inverse_transform(data[:, i]) for i, scaler in enumerate(self.scalers)])
    
class PercentileScaler:
    """Maps the 10th percentile -> 0 and the 90th percentile -> 1, per column."""
    def __init__(self, p10, p90, eps=1e-8):
        self.p10 = np.asarray(p10)
        self.range = np.asarray(p90) - self.p10
        self.range = np.where(self.range < eps, eps, self.range)  # avoid divide-by-zero on flat DOFs

    def transform(self, data):
        return (data - self.p10) / self.range

    def inverse_transform(self, data):
        return data * self.range + self.p10
    
class BehaviorScalerRank:
    def __init__(self, scaler):
        self.scaler = scaler
    def transform(self, data):
        return self.scaler.inverse_transform(data)
    def inverse_transform(self, data):
        return self.scaler.transform(data)

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
