import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import unittest
from neuraldecoding.model.neural_network_models import TCN
from neuraldecoding.utils.data_tools import add_history

class testTcFNN(unittest.TestCase):
    def setUp(self):
        seed=42
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.X = torch.randn(200, 2)
        self.y = torch.zeros((200, 1))
        for t in range(200):
            if t == 0:
                self.y[t] = torch.sum(self.X[t])
            else:
                self.y[t] = torch.sum(self.X[t]) + 0.5 * self.y[t-1]
        self.X = add_history(self.X.numpy(), 4)

        self.X_test = torch.randn(10, 2)
        self.y_test = torch.zeros((10, 1))
        for t in range(10):
            if t == 0:
                self.y_test[t] = torch.sum(self.X_test[t])
            else:
                self.y_test[t] = torch.sum(self.X_test[t]) + 0.5 * self.y_test[t-1]
        self.X_test = add_history(self.X_test.numpy(), 4)

        self.params = {
            "input_size": 2,
            "conv_size": 4,
            "conv_size_out": 8,
            "num_states": 1,
            "layer_size_list": [256, 256, 256, 256],
            "dropout_p": 0.5,
            "denormalize": True
        }

    def test_lstm_train_pred(self):
        model = TCN(params=self.params)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        loss_func = torch.nn.MSELoss()
        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            model.train_step(self.X, self.y, optimizer, loss_func) 
        model.eval()
        predictions = model(self.X_test)
        mse = loss_func(predictions, self.y_test)
        self.assertLess(mse.item(), 5)
    
    def test_save_and_load_model(self):
        model = TCN(params=self.params)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        loss_func = torch.nn.MSELoss()
        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            model.train_step(self.X, self.y, optimizer, loss_func) 
        model.eval()
        old_predictions = model(self.X_test)
        old_mse = loss_func(old_predictions, self.y_test)

        model_path = os.path.join("tests", "model", "cache", "tcFNN_model.pkl")
        model.save_model(model_path)

        loaded_model = TCN(params=self.params)
        loaded_model.load_model(model_path)
        loaded_model.eval()
        loaded_predictions = loaded_model(self.X_test)
        loaded_mse = loss_func(loaded_predictions, self.y_test)
        tolerance = 1e-1
        self.assertTrue(torch.allclose(loaded_predictions, old_predictions, atol=tolerance), "Predictions after loading do not match actual values")
        self.assertTrue(torch.allclose(loaded_mse, old_mse, atol=tolerance), "MSE after loading do not match actual values")
        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == "__main__":
    unittest.main()
