import unittest
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from neuraldecoding.trainer.NeuralNetworkTrainer import NNTrainer
from neuraldecoding.trainer.LinearTrainer import LinearTrainer
from neuraldecoding.utils import parse_verify_config
from neuraldecoding.preprocessing import Preprocessing
from hydra import initialize, compose

def load_config(cfg_name):
    """Loads the Hydra configuration file (train.yaml)."""
    cfg_path = os.path.join("..","configs",cfg_name)

    with initialize(version_base=None, config_path=cfg_path):
        config = compose("config")
    cfg = parse_verify_config(config, 'trainer')
    preprocessing_config = parse_verify_config(config, 'preprocessing')
    preprocessing_trainer_config = preprocessing_config['preprocessing_trainer']
    return cfg, preprocessing_config, preprocessing_trainer_config

class testNNTrainer(unittest.TestCase):
    def setUp(self):
        self.config, self.preprocessing_config, self.preprocessing_trainer_config = load_config("test_nn_trainer")
        preprocessor_trainer = Preprocessing(self.preprocessing_trainer_config)
        self.trainer = NNTrainer(preprocessor_trainer, self.config, {"data_path": "tests/trainer/data/sub-Monkey-N_ses-20200127_ecephys.nwb"})
        self.model, self.results = self.trainer.train_model()

    def test_results(self):
        self.assertIsInstance(self.results, dict, "Results should be a dict.")
        self.assertEqual(len(self.results['loss'][0]), 5)
        self.assertEqual(len(self.results['correlation'][0]), 5)
        self.assertEqual(len(self.results['loss'][1]), 5)
        self.assertEqual(len(self.results['correlation'][1]), 5)
    
    def test_model(self):
        self.assertIsInstance(self.model, torch.nn.Module, "Model should be an instance of torch.nn.Module.")

    def test_save(self):
        save_path = "tests/trainer/results/test_trainer_results.csv"
        self.assertTrue(os.path.exists(save_path), f"Results file {save_path} should exist after saving.")
        
        with open(save_path, 'r') as f:
            content = f.read()
            self.assertGreater(len(content), 0, "Results file should not be empty after saving.")

        os.remove(save_path)


class testLinearTrainer(unittest.TestCase):
    def setUp(self):
        self.config, self.preprocessing_config, self.preprocessing_trainer_config = load_config("test_linear_trainer")
        preprocessor_trainer = Preprocessing(self.preprocessing_trainer_config)
        self.trainer = LinearTrainer(preprocessor_trainer, self.config, {"data_path": "tests/trainer/data/sub-Monkey-N_ses-20200127_ecephys.nwb"})
        self.model, self.results = self.trainer.train_model()

    def test_results(self):
        self.assertIsInstance(self.results, dict, "Results should be a dict.")
        self.assertEqual(len(self.results['correlation'][0]), 1)
        self.assertEqual(len(self.results['correlation'][0][0]), 4)

    def test_save(self):
        save_path = "tests/trainer/results/test_trainer_results.csv"
        self.assertTrue(os.path.exists(save_path), f"Results file {save_path} should exist after saving.")
        
        with open(save_path, 'r') as f:
            content = f.read()
            self.assertGreater(len(content), 0, "Results file should not be empty after saving.")

        
        os.remove(save_path)

if __name__ == '__main__':
    unittest.main()
