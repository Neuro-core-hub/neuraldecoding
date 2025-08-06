import torch
from . import soft_dtw
from . import path_soft_dtw 

# Sophia's function for DILATE loss
def dilate_loss(outputs, targets, alpha, gamma, device, individuate_dofs=False):
    """
    Compute the DILATE loss for sequence alignment between `outputs` and `targets`.

    DILATE (DIstance, LATEncy) loss is designed for time-series alignment by combining:
    1. A shape-based term using differentiable soft-DTW.
    2. A temporal alignment penalty based on warping path deviation.

    Parameters
    ----------
    outputs : torch.Tensor
        Predicted sequences of shape (batch_size, N_output, T)

    targets : torch.Tensor
        Ground-truth sequences of shape (batch_size, N_output, T)

    alpha : float
        Weighting factor in [0, 1] to balance shape loss vs. temporal loss.

    gamma : float
        Soft-DTW smoothing parameter.

    device : torch.device
        Device on which tensors and computations are performed.

    Returns
    -------
    loss : torch.Tensor
        The combined DILATE loss.

    loss_shape : torch.Tensor
        The soft-DTW shape loss component.

    loss_temporal : torch.Tensor
        The temporal alignment penalty.
    """
    _, N, _ = outputs.shape
    pos_dim = N // 2

    if individuate_dofs:
        loss_list = []
        loss_shape_list = []
        loss_temporal_list = []
        for dof in range(pos_dim):
            dims_to_change = torch.Tensor([dof, dof+pos_dim]).to(int)
            relevant_outputs = outputs[:, dims_to_change, :]
            relevant_targets = targets[:, dims_to_change, :]
            loss, loss_shape, loss_temporal = dilate_func(relevant_outputs, relevant_targets, alpha, gamma, device)
            loss_list.append(loss)
            loss_shape_list.append(loss_shape)
            loss_temporal_list.append(loss_temporal)
        loss = sum(loss_list)
        loss_shape = sum(loss_shape_list)
        loss_temporal = sum(loss_temporal_list)
    else:
        loss, loss_shape, loss_temporal = dilate_func(outputs, targets, alpha, gamma, device)
    
    return loss, loss_shape, loss_temporal


def dilate_func(outputs, targets, alpha, gamma, device):
    """
    Compute the DILATE loss for sequence alignment between `outputs` and `targets`.

    DILATE (DIstance, LATEncy) loss is designed for time-series alignment by combining:
    1. A shape-based term using differentiable soft-DTW.
    2. A temporal alignment penalty based on warping path deviation.

    Parameters
    ----------
    outputs : torch.Tensor
        Predicted sequences of shape (batch_size, N_output, T)

    targets : torch.Tensor
        Ground-truth sequences of shape (batch_size, N_output, T)

    alpha : float
        Weighting factor in [0, 1] to balance shape loss vs. temporal loss.

    gamma : float
        Soft-DTW smoothing parameter.

    device : torch.device
        Device on which tensors and computations are performed.

    Returns
    -------
    loss : torch.Tensor
        The combined DILATE loss.

    loss_shape : torch.Tensor
        The soft-DTW shape loss component.

    loss_temporal : torch.Tensor
        The temporal alignment penalty.
    """
    batch_size, N_output, T = outputs.shape
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply

    # Compute distance matrices D for soft-DTW
    D = torch.zeros((batch_size, T, T), device=device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(
            targets[k].T.float(),  # shape (N_output, T) → time-major
            outputs[k].T.float()   # shape (N_output, T)
        )
        D[k] = Dk

    # Shape loss (soft-DTW)
    loss_shape = softdtw_batch(D, gamma)

    # Path-based temporal deviation penalty
    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)

    time_grid = torch.arange(0, T, dtype=torch.float32, device=device).view(-1, 1)
    Omega = soft_dtw.pairwise_distances(time_grid, time_grid)  # shape (T, T)

    loss_temporal = torch.sum(path * Omega) / (T * T)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal

    return loss, loss_shape, loss_temporal

"""
def dilate_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)
	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	return loss, loss_shape, loss_temporal
"""