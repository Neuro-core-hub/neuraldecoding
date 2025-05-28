import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from neuraldecoding.model.Model import Model
from neuraldecoding.trainer.Trainer import Trainer
from neuraldecoding.model.neural_network_models.NeuralNetworkModel import NeuralNetworkModel
from neuraldecoding.model.neural_network_models.LSTM import LSTM



@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(train_data, valid_data, config: DictConfig):
    """
    Trains a Neural Network. First loads data into dataloader, then uses TrainerImplementation class to train model.

    Parameters:
        train_data: training data, assumed numpy
        valid_data: validation data, assumed numpy
        config:  Hydra configuration file containing all required parameters (see train model function)
    
    Returns:
        model, results (tuple containing results information, see train model function)
    """
    trainer = TrainerImplementation()

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_data, dtype=torch.float32), dtype=torch.float32)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    
    model, results = trainer.train_model(train_dataloader, valid_dataloader, config)
    return model, results


class TrainerImplementation(Trainer):


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


    def train_model(self,
                    train_data: DataLoader,
                    valid_data: DataLoader,
                    config: DictConfig) -> Model:
        
        """
        Implements the training loop with the specified model and parameters.
        
        Parameters:
            train_data: Training data, should be torch DataLoader
            valid_data: Validation data, should be torch DataLoader
            config: Hydra configuration file containing all required parameters:
                - model
                - optimizer
                - scheduler
                - loss_function
                - training (dict containing: num_epochs, device, print_results (optional), print_every (optional))

        
        Returns:
            model, (loss_history_train, loss_history_val, corr_history): trained model,  training loss history, validation loss history, correlation history
        """
        
        # Main Training params
        model = self.create_model(config.model)
        optimizer = self.create_optimizer(config.optimizer, model.parameters())
        scheduler = self.create_scheduler(config.scheduler, optimizer)
        loss_func = self.create_loss_function(config.loss_func)

        # secondary training parameters
        num_epochs = config.training.num_epochs
        print_results = config.training.get("print_results", True)
        print_every = config.training.get("print_every", 10)

                
        # move to appropriate device       
        device = torch.device(config.training.device)
    
        # MAIN LOOP
        loss_history_train, loss_history_val, corr_history = [], [], []
        for epoch in range(num_epochs):

            # Train for one epoch
            train_loss = model._train_one_epoch(train_data, model, optimizer, loss_func, device)

            # Validate after each epoch
            val_loss, correlation = model._validate_one_epoch(valid_data, model, loss_func, device)

            # Record losses and correlation
            loss_history_train.append(train_loss)
            loss_history_val.append(val_loss)
            corr_history.append(correlation)

            # Scheduler step
            if scheduler:
                scheduler.step()

            # Print progress
            if print_results and (epoch % print_every == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch}/{num_epochs - 1}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Correlation: {correlation[0]:.4f}, {correlation[1]:.4f}")

            if print_results:
                print("*** Training Complete ***")

        return model, (loss_history_train, loss_history_val, corr_history)


