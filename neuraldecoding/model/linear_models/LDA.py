import numpy as np
import pickle
import torch
from typing import Any
from .LinearModel import LinearModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA(LinearModel):
    def __init__(self, params = {}):
        """
        Linear Discriminant Analysis (LDA) model for classification tasks.
        Args:
            params (dict): Parameters for the LDA model (sklearn.LinearDiscriminantAnalysis).
        """
        super().__init__(params)
        self.model = LinearDiscriminantAnalysis(**params)

    def __call__(self, data: Any) -> Any:
        """
        Wrapper function to make the instance callable.
        Args:
            data (torch.tensor or numpy.ndarray): Input data for the model (Nsamp, Nfeat).
        Returns:
            torch.tensor: Predictions from the model.
        """
        return self.forward(data)

    def train_step(self, input_data: Any) -> None:
        """
        Train the LDA model using the provided input data.
        Args:
            input_data (tuple): A tuple containing the input features (X, (Nsamp, Nfeat)) and target labels (y, (Nsamp)).
        """
        X, y = input_data
        self.model.fit(X, y)
        self.Nfeats = X.shape[1]

    def forward(self, data: Any) -> Any:
        """
        Predict outputs given input data.
        Args:
            data (torch.tensor or numpy.ndarray): Input data for prediction (Nsamp, Nfeat).
        Returns:
            torch.tensor: Predicted outputs from the model.
        """
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional.")
        if data.shape[1] != self.Nfeats:
            raise ValueError(f"Number of features in input data ({data.shape[1]}) does not match the model's trained number of features ({self.Nfeats}).")
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        predictions = self.model.predict(data)
        return torch.tensor(predictions, dtype=torch.float64)
    
    def get_likelihood(self, data: Any) -> Any:
        """
        Get the likelihood of the input data given the model.
        Args:
            data (torch.tensor or numpy.ndarray): Input data for likelihood estimation (Nsamp, Nfeat).
        Returns:
            torch.tensor: Likelihood estimates from the model.
        """
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional.")
        if data.shape[1] != self.Nfeats:
            raise ValueError(f"Number of features in input data ({data.shape[1]}) does not match the model's trained number of features ({self.Nfeats}).")
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        likelihood = self.model.predict_proba(data)
        return torch.tensor(likelihood, dtype=torch.float64)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained LDA model to a file.
        Args:
            filepath (str): Path to the file where the model will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained LDA model from a file.
        Args:
            filepath (str): Path to the file from which the model will be loaded.
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)