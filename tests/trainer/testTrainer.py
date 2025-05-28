import unittest
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from neuraldecoding.trainer.NeuralNetworkTrainer import TrainerImplementation

def load_synthetic_data():
    """Loads the synthetic train and validation data from .npy files."""
    data_train = np.load(os.path.join("tests", "trainer", "data", "train.npz"), allow_pickle=True)
    train_X, train_Y = data_train["X"], data_train["Y"]
    
    data_valid = np.load(os.path.join("tests", "trainer", "data", "valid.npz"), allow_pickle=True)
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
    return OmegaConf.load(os.path.join("configs", "trainer", "testTrainer.yaml"))

class testTrainer(unittest.TestCase):
    def setUp(self):
        """Set up the trainer and dataloaders for testing."""
        train_X, train_Y, valid_X, valid_Y = load_synthetic_data()
        self.train_loader, self.valid_loader = create_dataloaders(train_X, train_Y, valid_X, valid_Y)
        self.config = load_config()
        self.trainer = TrainerImplementation()

    def test_training_loop(self):
        """Tests the full training process to ensure it completes successfully."""
        model, results = self.trainer.train_model(self.train_loader, self.valid_loader, self.config)
        self.assertIsInstance(model, torch.nn.Module, "Output should be a trained model.")
        self.assertIsInstance(results, tuple, "Results should be a tuple.")
        self.assertEqual(len(results), 3, "Results should include loss history and correlation history.")

if __name__ == '__main__':
    unittest.main()
