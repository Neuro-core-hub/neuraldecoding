import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import torch
import unittest
from neuraldecoding.utils.data_tools import add_history
try:
    import einops
    from neuraldecoding.model.neural_network_models import TransformerModel, TransformerGRUModel
except ImportError:
    raise unittest.SkipTest("Required dependencies einops (for Transformer) not available. Test for transformer skipped")

class testTransformer(unittest.TestCase):
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

    def test_Transformer_train_pred(self):
        params = {
            'num_features': 2,
            'num_outputs': 1,
            'enc_nhead': 2,
            'enc_nhid': 2048,
            'enc_nlayers': 1,
            'dropout': 0.5
        }
        model = TransformerModel(params=params)
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
    
    def test_TransformerGRU_train_pred(self):
        params = {
            'num_features': 2,
            'num_outputs': 1,
            'enc_nhead': 2,
            'enc_nhid': 2048,
            'enc_nlayers': 1,
            'dropout': 0.5,
            'rnn_nhid': 300
        }
        model = TransformerGRUModel(params=params)
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

    # def test_Conformer_train_pred(self):
    #     params = {
    #         'num_channels': 2,
    #         'num_outputs': 1,
    #         'good_channels_idx': [0, 1],
    #         'emb_size': 40,
    #         'num_heads': 10,
    #         'num_layers': 6,
    #         'drop_prob': 0.5
    #     }
    #     model = ConformerModel(params=params)
    #     optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    #     loss_func = torch.nn.MSELoss()
    #     model.train()
    #     for epoch in range(20):
    #         optimizer.zero_grad()
    #         model.train_step(self.X, self.y, optimizer, loss_func) 
    #     model.eval()
    #     predictions = model(self.X_test)
    #     mse = loss_func(predictions, self.y_test)
    #     self.assertLess(mse.item(), 5)
    
    def test_Transformer_save_and_load_model(self):
        params = {
            'num_features': 2,
            'num_outputs': 1,
            'enc_nhead': 2,
            'enc_nhid': 2048,
            'enc_nlayers': 1,
            'dropout': 0.5
        }
        model = TransformerModel(params=params)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        loss_func = torch.nn.MSELoss()
        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            model.train_step(self.X, self.y, optimizer, loss_func) 
        model.eval()
        old_predictions = model(self.X_test)
        old_mse = loss_func(old_predictions, self.y_test)

        model_path = os.path.join("tests", "model", "cache", "transformer_model.pkl")
        model.save_model(model_path)

        loaded_model = TransformerModel(params=params)
        loaded_model.load_model(model_path)
        loaded_model.eval()
        loaded_predictions = loaded_model(self.X_test)
        loaded_mse = loss_func(loaded_predictions, self.y_test)
        tolerance = 1e-1

        self.assertTrue(torch.allclose(loaded_predictions, old_predictions, atol=tolerance), "Predictions after loading do not match actual values")
        self.assertTrue(torch.allclose(loaded_mse, old_mse, atol=tolerance), "MSE after loading do not match actual values")
        if os.path.exists(model_path):
            os.remove(model_path)

    def test_TransformerGRU_save_and_load_model(self):
        params = {
            'num_features': 2,
            'num_outputs': 1,
            'enc_nhead': 2,
            'enc_nhid': 2048,
            'enc_nlayers': 1,
            'dropout': 0.5,
            'rnn_nhid': 300
        }
        model = TransformerGRUModel(params=params)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
        loss_func = torch.nn.MSELoss()
        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            model.train_step(self.X, self.y, optimizer, loss_func) 
        model.eval()
        old_predictions = model(self.X_test)
        old_mse = loss_func(old_predictions, self.y_test)

        model_path = os.path.join("tests", "model", "cache", "transformergru_model.pkl")
        model.save_model(model_path)

        loaded_model = TransformerGRUModel(params=params)
        loaded_model.load_model(model_path)
        loaded_model.eval()
        loaded_predictions = loaded_model(self.X_test)
        loaded_mse = loss_func(loaded_predictions, self.y_test)
        tolerance = 1e-1

        self.assertTrue(torch.allclose(loaded_predictions, old_predictions, atol=tolerance), "Predictions after loading do not match actual values")
        self.assertTrue(torch.allclose(loaded_mse, old_mse, atol=tolerance), "MSE after loading do not match actual values")
        if os.path.exists(model_path):
            os.remove(model_path)
    
    # def test_Conformer_save_and_load_model(self):
    #     params = {
    #         'num_channels': 2,
    #         'num_outputs': 1,
    #         'good_channels_idx': [0, 1],
    #         'emb_size': 40,
    #         'num_heads': 10,
    #         'num_layers': 6,
    #         'drop_prob': 0.5
    #     }
    #     model = ConformerModel(params=params)
    #     optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    #     loss_func = torch.nn.MSELoss()
    #     model.train()
    #     for epoch in range(20):
    #         optimizer.zero_grad()
    #         model.train_step(self.X, self.y, optimizer, loss_func) 
    #     model.eval()
    #     old_predictions = model(self.X_test)
    #     old_mse = loss_func(old_predictions, self.y_test)

    #     model_path = os.path.join("tests", "model", "cache", "conformer_model.pkl")
    #     model.save_model(model_path)

    #     loaded_model = ConformerModel(params=params)
    #     loaded_model.load_model(model_path)
    #     loaded_model.eval()
    #     loaded_predictions = loaded_model(self.X_test)
    #     loaded_mse = loss_func(loaded_predictions, self.y_test)
    #     tolerance = 1e-1

    #     self.assertTrue(torch.allclose(loaded_predictions, old_predictions, atol=tolerance), "Predictions after loading do not match actual values")
    #     self.assertTrue(torch.allclose(loaded_mse, old_mse, atol=tolerance), "MSE after loading do not match actual values")
    #     if os.path.exists(model_path):
    #         os.remove(model_path)

if __name__ == "__main__":
    unittest.main()
