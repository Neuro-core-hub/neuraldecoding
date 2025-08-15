import numpy as np
import torch
from sklearn.metrics import r2_score
from neuraldecoding.utils.loss_functions import soft_dtw, path_soft_dtw, dilate_loss

def mse(pred, target, params=None):
    """Calculates the mean squared error between y1 and y2 (tensors)"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    if len(pred) != len(target):
        raise ValueError("pred and target must have the same length")
    
    if len(pred) == 0:
        raise ValueError("Arrays cannot be empty")
    
    return np.mean((pred - target) ** 2, axis=0)

def dilate(pred, target, params={'per_dof': True, 'alpha': 0.5, 'gamma': 0.001, 'device':'cpu'}):
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred, device=params['device'])
    if isinstance(target, np.ndarray):
        target = torch.tensor(target, device=params['device'])

    if params['per_dof']:
        pos_dim = pred.shape[1] // 2
        metrics = np.zeros((pos_dim))
        for dof in range(pos_dim):
            dims_to_measure = torch.Tensor([dof, dof+pos_dim]).to(int)
            pred_dof = pred[:, dims_to_measure]
            target_dof = target[:, dims_to_measure]
            metrics[dof] = dilate_loss(pred_dof, target_dof, alpha=params['alpha'], gamma=params['gamma'], device=pred.device)
    else:
        metrics = dilate_loss(pred, target, alpha=params['alpha'], gamma=params['gamma'], device=pred.device)

    return metrics

def softdtw(pred, target, params={'per_dof': True, 'gamma': 0.001, 'device':'cpu'}): 
    """Calculates the DILATE loss between y1 and y2 (tensors)"""
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred, device=params['device'])
    if isinstance(target, np.ndarray):
        target = torch.tensor(target, device=params['device'])

    T = pred.shape[0]
    N = pred.shape[1]
    pred = pred.transpose(0, 1).unsqueeze(0)  # shape (1, N_output, T)
    target = target.transpose(0, 1).unsqueeze(0)  # shape (1, N_output, T))

    pos_dim = N // 2

    softdtw_func = soft_dtw.SoftDTWBatch.apply
    if params['per_dof']:
        metrics = np.zeros((pos_dim))
        for dof in range(pos_dim):
            dims_to_measure = torch.Tensor([dof, dof+pos_dim]).to(int)
            pred_dof = pred[0, dims_to_measure, :].unsqueeze(0)  # shape (1, N_output, T)
            target_dof = target[0, dims_to_measure, :].unsqueeze(0)  # shape (1, N_output, T)
            D = torch.zeros((1, T, T), device=params['device'])
            Dk = soft_dtw.pairwise_distances(
                target_dof[0].T.float(),  # shape (N_output, T) → time-major
                pred_dof[0].T.float()   # shape (N_output, T)
            )
            D[0] = Dk
            metrics[dof] = softdtw_func(D, params['gamma'])
    else:
        D = torch.zeros((1, T, T), device=params['device'])
        Dk = soft_dtw.pairwise_distances(
            target[0].T.float(),  # shape (N_output, T) → time-major
            pred[0].T.float()   # shape (N_output, T)
        )
        D[0] = Dk
        metrics = softdtw_func(D, params['gamma'])
    
    return metrics

def tdi(pred, target, params={'per_dof': True, 'gamma': 0.001, 'device':'cpu'}): 
    """Calculates the DILATE loss between y1 and y2 (tensors)"""
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred, device=params['device'])
    if isinstance(target, np.ndarray):
        target = torch.tensor(target, device=params['device'])

    T = pred.shape[0]
    N = pred.shape[1]

    pos_dim = N // 2
    
    pred = pred.transpose(0, 1).unsqueeze(0)
    target = target.transpose(0, 1).unsqueeze(0)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    
    if params['per_dof']:
        metrics = np.zeros((pos_dim))
        for dof in range(pos_dim):
            dims_to_measure = torch.Tensor([dof, dof+pos_dim]).to(int)
            pred_dof = pred[0, dims_to_measure, :].unsqueeze(0)  # shape (1, N_output, T)
            target_dof = target[0, dims_to_measure, :].unsqueeze(0)  # shape (1, N_output, T)
            D = torch.zeros((1, T, T), device=params['device'])
            Dk = soft_dtw.pairwise_distances(
                pred_dof[0].T.float(),  # shape (1, T) → time-major
                target_dof[0].T.float()   # shape (1, T)
            )
            D[0] = Dk
            path = path_dtw(D, params['gamma'])

            pred_dof = pred_dof.transpose(0, 1).unsqueeze(0)  # shape (1, N_output, T)
            target_dof = target_dof.transpose(0, 1).unsqueeze(0)  # shape (1, N_output, T))

            time_grid = torch.arange(0, T, dtype=torch.float32, device=params['device']).view(-1, 1)
            Omega = soft_dtw.pairwise_distances(time_grid, time_grid)  # shape (T, T)

            metrics[dof] = torch.sum(path * Omega) / (T * T)
    else:
        D = torch.zeros((1, T, T), device=params['device'])
        Dk = soft_dtw.pairwise_distances(
            pred[0].T.float(),  # shape (N_output, T) → time-major
            target[0].T.float()   # shape (N_output, T)
        )
        D[0] = Dk

        path = path_dtw(D, params['gamma'])

        time_grid = torch.arange(0, T, dtype=torch.float32, device=params['device']).view(-1, 1)
        Omega = soft_dtw.pairwise_distances(time_grid, time_grid)  # shape (T, T)
        metrics = torch.sum(path * Omega) / (T * T)
    return metrics

def correlation(pred,target,params=None):
    """Calculates the correlation between y1 and y2 (tensors)"""
    corr = []
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    for i in range(pred.shape[1]):
        corr.append(np.corrcoef(pred[:, i], target[:, i])[1, 0])
    return corr

def accuracy(pred, target, params=None):
    if len(pred) != len(target):
        raise ValueError("pred and target must have the same length")
    
    if len(pred) == 0:
        raise ValueError("Arrays cannot be empty")
    
    correct_predictions = (pred == target).sum()
    total_predictions = len(pred)
    return correct_predictions / total_predictions

def r2(pred, target, params=None):
    if len(pred) != len(target):
        raise ValueError("pred and target must have the same length")

    if len(pred) == 0:
        raise ValueError("Arrays cannot be empty")

    return r2_score(target, pred, **params) if params else r2_score(target, pred)