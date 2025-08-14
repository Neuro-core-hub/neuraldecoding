import numpy as np
import torch
from sklearn.metrics import r2_score

def correlation(pred,target,params=None):
    """Calculates the correlation between y1 and y2 (tensors)"""
    corr = []
    for i in range(pred.shape[1]):
        corr.append(np.corrcoef(pred.squeeze()[:, i], target.squeeze()[:, i])[1, 0])
    return corr

def accuracy(pred, target, params=None):
    if len(pred) != len(target):
        raise ValueError("pred and target must have the same length")
    
    if len(pred) == 0:
        raise ValueError("Arrays cannot be empty")
    
    correct_predictions = (pred.squeeze() == target.squeeze()).sum()
    total_predictions = len(pred)
    return correct_predictions / total_predictions

def r2(pred, target, params=None):
    if len(pred) != len(target):
        raise ValueError("pred and target must have the same length")

    if len(pred) == 0:
        raise ValueError("Arrays cannot be empty")

    return r2_score(target, pred, **params) if params else r2_score(target, pred)

def mse(pred, target, params=None):
    """Calculates the mean squared error between predictions and targets"""
    if len(pred) != len(target):
        raise ValueError("pred and target must have the same length")
    
    if len(pred) == 0:
        raise ValueError("Arrays cannot be empty")
    
    mse_values = []
    for i in range(pred.shape[1]):
        mse_values.append(np.mean((pred[:, i] - target[:, i]) ** 2))
    return mse_values
