import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import unittest
from neuraldecoding.model.neural_network_models import RecurrentModel
from neuraldecoding.utils.data_tools import add_history
class testRNN(unittest.TestCase):
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
        self.X = add_history(self.X.numpy(), 3)

        self.X_test = torch.randn(10, 2)
        self.y_test = torch.zeros((10, 1))
        for t in range(10):
            if t == 0:
                self.y_test[t] = torch.sum(self.X_test[t])
            else:
                self.y_test[t] = torch.sum(self.X_test[t]) + 0.5 * self.y_test[t-1]
        self.X_test = add_history(self.X_test.numpy(), 3)

        self.params = {
            "input_size": 2,
            "hidden_size": 10,
            "num_outputs": 1,
            "num_layers": 1,
            "rnn_type": "lstm",
            "drop_prob": 0,
            "hidden_noise_std": None,
            "dropout_input": 0
        }

    def test_lstm_train_pred(self):
        self.params['rnn_type'] = 'lstm'
        model = RecurrentModel(params=self.params)
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
    
    def test_rnn_train_pred(self):
        self.params['rnn_type'] = 'rnn'
        model = RecurrentModel(params=self.params)
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

    def test_gru_train_pred(self):
        self.params['rnn_type'] = 'gru'
        model = RecurrentModel(params=self.params)
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
        self.params['rnn_type'] = 'lstm'
        model = RecurrentModel(params=self.params)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        loss_func = torch.nn.MSELoss()
        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            model.train_step(self.X, self.y, optimizer, loss_func) 
        model.eval()
        old_predictions = model(self.X_test)
        old_mse = loss_func(old_predictions, self.y_test)

        model_path = os.path.join("tests", "model", "cache", "rnn_model.pkl")
        model.save_model(model_path)

        loaded_model = RecurrentModel(params=self.params)
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
