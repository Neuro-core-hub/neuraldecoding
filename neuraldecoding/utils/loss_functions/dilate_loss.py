import torch
from . import soft_dtw
from . import path_soft_dtw 

class DILATELoss:
    def __init__(self, alpha=0.5, gamma=0.001, device='cuda', individuate_dofs=False, only_final_loss=True):
        self.alpha = alpha
        self.gamma = gamma
        self.device = torch.device(device)
        self.individuate_dofs = individuate_dofs
        self.only_final_loss = only_final_loss

    def __call__(self, outputs, targets):
        return self.dilate_loss(outputs, targets)

    def dilate_loss(self, outputs, targets):
        # Ensure batch dimension
        if outputs.ndim == 2:
            outputs = outputs.transpose(0, 1).unsqueeze(0)
            targets = targets.transpose(0, 1).unsqueeze(0)

        _, N, _ = outputs.shape
        pos_dim = N // 2

        if self.individuate_dofs:
            loss_list, loss_shape_list, loss_temporal_list = [], [], []
            for dof in range(pos_dim):
                dims_to_change = torch.tensor([dof, dof + pos_dim], dtype=torch.long)
                relevant_outputs = outputs[:, dims_to_change, :]
                relevant_targets = targets[:, dims_to_change, :]
                l, ls, lt = self.dilate_func(relevant_outputs, relevant_targets)
                loss_list.append(l)
                loss_shape_list.append(ls)
                loss_temporal_list.append(lt)
            loss = sum(loss_list)
            loss_shape = sum(loss_shape_list)
            loss_temporal = sum(loss_temporal_list)
        else:
            loss, loss_shape, loss_temporal = self.dilate_func(outputs, targets)

        if self.only_final_loss:
            return loss
        return loss, loss_shape, loss_temporal

    def dilate_func(self, outputs, targets):
        batch_size, N_output, T = outputs.shape
        device = self.device
        alpha, gamma = self.alpha, self.gamma

        D = torch.zeros((batch_size, T, T), device=device)
        for k in range(batch_size):
            D[k] = soft_dtw.pairwise_distances(
                targets[k].T.float(), outputs[k].T.float()
            )

        loss_shape = soft_dtw.SoftDTWBatch.apply(D, gamma)

        path = path_soft_dtw.PathDTWBatch.apply(D, gamma)

        time_grid = torch.arange(0, T, dtype=torch.float32, device=device).view(-1, 1)
        Omega = soft_dtw.pairwise_distances(time_grid, time_grid)

        loss_temporal = torch.sum(path * Omega) / (T * T)
        loss = alpha * loss_shape + (1 - alpha) * loss_temporal

        return loss, loss_shape, loss_temporal