import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import unittest
from neuraldecoding.model.linear_models import RidgeRegression

class testRidgeRegression(unittest.TestCase):
    def setUp(self):
        seed=42
        np.random.seed(seed)
        self.X = np.random.rand(100, 1)
        self.y = 3 * self.X.squeeze() + 2 + np.random.randn(100) * 0.1
        self.params = {"fit_intercept": True}
        self.model = RidgeRegression(params=self.params)

    def test_train_and_predict(self):
        self.model.train_step((self.X, self.y))

        X_test = np.array([[0.5], [1.5]])
        predictions = self.model(X_test)
        actual_predictions = torch.tensor([3.48949705, 6.14089983], dtype=torch.float64)
        tolerance = 1e-1
        self.assertTrue(torch.allclose(predictions, actual_predictions, atol=tolerance), "Predictions do not match actual values")

    def test_save_and_load_model(self):
        self.model.train_step((self.X, self.y))

        model_path = os.path.join("tests", "model", "cache", "ridge_regression_model.pkl")
        self.model.save_model(model_path)

        loaded_model = RidgeRegression(params=self.params)
        loaded_model.load_model(model_path)

        X_test = np.array([[0.5], [1.5]])
        predictions_loaded = loaded_model(X_test)
        actual_predictions = torch.tensor([3.48949705, 6.14089983], dtype=torch.float64)
        tolerance = 1e-1

        self.assertTrue(torch.allclose(predictions_loaded, actual_predictions, atol=tolerance), "Predictions after loading do not match actual values")

        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == "__main__":
    unittest.main()
