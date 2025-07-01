import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
from neuraldecoding.decoder import LinearDecoder
from hydra import initialize, compose
from neuraldecoding.utils import parse_verify_config

class TestLinearDecoder(unittest.TestCase):
    def setUp(self):
        cfg_path = os.path.join("..","..","configs","test_linear_decoder")

        with initialize(version_base=None, config_path=cfg_path):
            config = compose("config")
        self.cfg = parse_verify_config(config, 'decoder')
        self.decoder = LinearDecoder(self.cfg)

    def test_initialization(self):
        np.testing.assert_array_equal(self.decoder.get_input_shape(), np.array([2, 1]))
        np.testing.assert_array_equal(self.decoder.get_output_shape(), np.array([2]))
        self.assertEqual(
            os.path.normpath(self.decoder.fpath), 
            os.path.join("tests", "decoder", "models", "linear_regression_model.pkl")
        )

    def test_load_model(self):
        self.decoder.load_model()

    def test_predict(self):
        X_test = np.array([[0.5], [1.5]])
        prediction = self.decoder.predict(X_test)
        actual_predictions =torch.tensor([3.49852095, 6.45254363], dtype=torch.float64)

        tolerance = 1e-1
        self.assertTrue(torch.allclose(prediction, actual_predictions, atol=tolerance), "Predictions after loading do not match actual values")
        
    def test_get_input_shape(self):
        np.testing.assert_array_equal(self.decoder.get_input_shape(), np.array([2, 1]))

    def test_get_output_shape(self):
        np.testing.assert_array_equal(self.decoder.get_output_shape(), np.array([2,]))


if __name__ == "__main__":
    unittest.main()