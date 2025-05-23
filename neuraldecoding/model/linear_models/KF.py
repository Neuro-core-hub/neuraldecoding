import numpy as np
import pickle
import torch
from tqdm import tqdm
from LinearModel import LinearModel


class KalmanFilter(LinearModel):
    def __init__(self, model_params):
        """
        Constructs a Kalman Filter decoder object (can use numpy arrays, torch arrays, or torch dataloaders). 
        
        Parameters:
            model_params (dict) containing three keys, 'append_ones_y' (bool) which specifies whether or not to add column of ones for bias, 
            'device' (bool) which specifies what device to train/run the model on and "return_tensor" 
            (bool) which specifies whether a tensor should be returned or not.

            TODO: option to zero position uncertainty
            TODO: make it store the current state (currently resets every time)
            TODO: not tested on GPU
        """

        self.A, self.C, self.W, self.Q = None, None, None, None
        self.At, self.Ct = None, None
        self.append_ones_y = model_params["append_ones_y"]
        self.device = model_params["device"]
        self.return_tensor = model_params["return_tensor"]

    def __call__(self, data):
        """
        Makes the instance callable and returns the result of forward pass.

        Parameters:
            data (ndarray): Observation data for prediction, expected size [n, m]

        Returns:
            ndarray: Prediction results, size [n, k]
        """
        return self.forward(data)
    

    def train_step(self, input_data):
        """
        Trains the matrices in the model. If append_ones_y is true, a column of ones is added to calculate bias. 

        Parameters:
            input_data (tuple): A tuple of (x, y)
                - x (ndarray) size [n, m]: observation features
                - y (ndarray) size [n, k]: hidden state features

        """

        # unpack input data
        x, y = input_data

        if self.A is not None:
            raise ValueError("Tried to train a model that's already trained ")
        
        if self.append_ones_y:
            y = torch.cat((y, torch.ones([y.shape[0], 1])), dim=1)

        ytm1 = y[:-1, :]
        yt = y[1:, :]

        self.A = (yt.T @ ytm1) @ torch.pinverse(ytm1.T @ ytm1)                              # kinematic trajectory model
        self.W = (yt - (ytm1 @ self.A.T)).T @ (yt - (ytm1 @ self.A.T)) / (yt.shape[0] - 1)  # trajectory model noise
        self.C = (x.T @ y) @ torch.pinverse(y.T @ y)                                        # neural observation model
        self.Q = (x - (y @ self.C.T)).T @ (x - (y @ self.C.T)) / yt.shape[0]                # observation model noise

        self.At = self.A.T
        self.Ct = self.C.T

    def forward(self, input):
        """
        Runs a forward pass, by calling a predict method (torch or numpy).

        Parameters:
            input (ndarray) size [n, m], where n is the number of samples/data, and m is the number of observation features
        
        Returns:
            yhat (ndarray) prediction of size [n, k], where n is the number of samples/data, and k is the number of hidden state features
        """
        yhat = self.predict_numpy(input)
        return yhat

    def predict_numpy(self, x, start_y = None):
        """
        Runs a forward pass, returning a prediction for all input datapoints. If start_y is true, initial state is added.

        Parameters:
            input (ndarray) size [n, m], where n is the number of samples/data, and m is the number of observation features
            start_y (array, optional) which specifies an initial [1, m] state.
        
        Returns:
            yhat (ndarray) prediction of size [n, k], where n is the number of samples/data, and k is the number of hidden state features
        """
        x = x.view((x.shape[0], -1))
        x = x.numpy()

        self.A = self.A.numpy()
        self.W = self.W.numpy()
        self.C = self.C.numpy()
        self.Q = self.Q.numpy()

        yhat = np.zeros((x.shape[0], self.A.shape[1]))
        if start_y:
            yhat[0, :] = start_y

        Pt = self.W.copy()

        for t in tqdm(range(1, yhat.shape[0])):
            yt = yhat[t-1, :] @ self.A.T                                # predict new state
            Pt = self.A @ Pt @ self.A.T + self.W                        # compute error covariance
            K = np.linalg.lstsq((self.C @ Pt @ self.C.T + self.Q).T,
                                (Pt @ self.C.T).T, rcond=None)[0].T     # compute kalman gain, where B/A = (A'\B')'
            yhat[t, :] = yt.T + K @ (x[t, :].T - self.C @ yt.T)	        # update state estimate
            Pt = (np.eye(Pt.shape[0]) - K @ self.C) @ Pt	            # update error covariance

        if self.return_tensor:
            yhat = torch.Tensor(yhat)
        if self.append_ones_y:
            yhat = yhat[:, :-1]
        return yhat

    def save_model(self, fpath):
        """
        Saves the model in its current state at the specified filepath

        Parameters:
            filepath (path-like object) indicates the file path to save the model in
        """
        model_dict = {
            "A": self.A,
            "C": self.C,
            "W": self.W,
            'Q': self.Q,
            "Model": "KF"
        }

        with open(fpath, "wb") as file:
            pickle.dump(model_dict, file)

    def load_model(self, fpath):
        """
        Load model parameters from a specified location

        Parameters:
            filepath (path-like object) indicates the file path to load the model from
        """

        with open(fpath, "rb") as file:
           model_dict = pickle.load(file)

        if model_dict["Model"] != "KF":
            raise Exception("Tried to load model that isn't a Kalman Filter Instance")
        
        self.A = model_dict['A']
        self.C = model_dict['C']
        self.W = model_dict['W']
        self.Q = model_dict['Q']
