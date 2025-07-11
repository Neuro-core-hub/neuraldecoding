import unittest
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from neuraldecoding.trainer.NeuralNetworkTrainer import NNTrainer
from neuraldecoding.utils import parse_verify_config
from neuraldecoding.preprocessing import Preprocessing
from hydra import initialize, compose

def load_config():
    """Loads the Hydra configuration file (train.yaml)."""
    cfg_path = os.path.join("..","..","configs","test_new_trainer")

    with initialize(version_base=None, config_path=cfg_path):
        config = compose("config")
    cfg = parse_verify_config(config, 'trainer')
    preprocessing_config = parse_verify_config(config, 'preprocessing')
    preprocessing_trainer_config = preprocessing_config['preprocessing_trainer']
    return cfg, preprocessing_config, preprocessing_trainer_config

class testTrainer(unittest.TestCase):
    def setUp(self):
        """Set up the trainer and dataloaders for testing."""
        self.config, self.preprocessing_config, self.preprocessing_trainer_config = load_config()
        preprocessor_trainer = Preprocessing(self.preprocessing_trainer_config)
        self.trainer = NNTrainer(preprocessor_trainer, self.config)

    def test_training_loop(self):
        """Tests the full training process to ensure it completes successfully."""
        model, results = self.trainer.train_model()
        self.assertIsInstance(model, torch.nn.Module, "Output should be a trained model.")
        self.assertIsInstance(results, dict, "Results should be a dict.")

if __name__ == '__main__':
    unittest.main()
