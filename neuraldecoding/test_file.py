import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from neuraldecoding.trainer.NeuralNetworkTrainer import TrainerImplementation

def load_synthetic_data():
    """Loads the synthetic train and validation data from .npy files."""
    data_train = np.load("./dataset/train.npz", allow_pickle=True)
    train_X, train_Y = data_train["X"], data_train["Y"]
    
    data_valid = np.load("./dataset/valid.npz", allow_pickle=True)
    valid_X, valid_Y = data_valid["X"], data_valid["Y"]
    
    return train_X, train_Y, valid_X, valid_Y
def create_dataloaders(train_X, train_Y, valid_X, valid_Y, batch_size=32):
    """Creates PyTorch DataLoaders for training and validation data."""
    train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), 
                                  torch.tensor(train_Y, dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(valid_X, dtype=torch.float32), 
                                  torch.tensor(valid_Y, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

def load_config():
    """Loads the Hydra configuration file (train.yaml)."""
    return OmegaConf.load("configs/train.yaml")

@pytest.fixture
def setup_trainer():
    """Fixture to set up the trainer and dataloaders for testing."""
    train_X, train_Y, valid_X, valid_Y = load_synthetic_data()
    train_loader, valid_loader = create_dataloaders(train_X, train_Y, valid_X, valid_Y)
    config = load_config()
    trainer = TrainerImplementation()
    return trainer, train_loader, valid_loader, config

def test_train_one_epoch(setup_trainer):
    """Tests one epoch of training to ensure loss decreases."""
    trainer, train_loader, _, config = setup_trainer
    model = trainer.create_model(config.model)
    optimizer = trainer.create_optimizer(config.optimizer, model.parameters())
    loss_func = trainer.create_loss_function(config.loss_func)
    device = torch.device(config.training.device)
    model.to(device)
    initial_loss = trainer._train_one_epoch(train_loader, model, optimizer, loss_func, device)
    new_loss = trainer._train_one_epoch(train_loader, model, optimizer, loss_func, device)
    assert new_loss <= initial_loss, "Training loss should decrease after one epoch."

def test_validate_one_epoch(setup_trainer):
    """Tests validation loss computation and correlation metric calculation."""
    trainer, _, valid_loader, config = setup_trainer
    model = trainer.create_model(config.model)
    loss_func = trainer.create_loss_function(config.loss_func)
    device = torch.device(config.training.device)
    model.to(device)
    val_loss, correlation = trainer._validate_one_epoch(valid_loader, model, loss_func, device)
    assert val_loss >= 0, "Validation loss should be non-negative."
    assert isinstance(correlation, list) and all(-1 <= c <= 1 for c in correlation), "Correlation values should be between -1 and 1."

def test_training_loop(setup_trainer):
    """Tests the full training process to ensure it completes successfully."""
    trainer, train_loader, valid_loader, config = setup_trainer
    model, results = trainer.train_model(train_loader, valid_loader, config)
    assert isinstance(model, torch.nn.Module), "Output should be a trained model."
    assert isinstance(results, tuple) and len(results) == 3, "Results should include loss history and correlation history."
