import unittest
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from neuraldecoding.trainer.NeuralNetworkTrainer import LSTMTrainer
from neuraldecoding.utils import parse_verify_config
from hydra import initialize, compose

def load_config():
    """Loads the Hydra configuration file (train.yaml)."""
    cfg_path = os.path.join("..","..","configs","test_new_trainer")

    with initialize(version_base=None, config_path=cfg_path):
        config = compose("config")
    cfg = parse_verify_config(config, 'trainer')
    return cfg

class testTrainer(unittest.TestCase):
    def setUp(self):
        """Set up the trainer and dataloaders for testing."""
        self.config = load_config()
        self.trainer = LSTMTrainer(self.config)

    def test_training_loop(self):
        """Tests the full training process to ensure it completes successfully."""
        model, logger = self.trainer.train_model()
        self.assertIsInstance(model, torch.nn.Module, "Output should be a trained model.")
        self.assertIsInstance(logger, dict, "Results should be a tuple.")
        self.assertEqual(len(logger.keys()), 3, "Results should include loss history and correlation history.")

if __name__ == '__main__':
    unittest.main()
