import torch
import numpy as np
import warnings

class RankLoss:
    def __init__(self, transition_time = 5, lambda_flat_active = 100, lambda_flat_inactive = 400, device='cuda'):
        """
        
        :param self: RankLoss instance
        :param transition_time: number of time steps after onset to apply flatness loss
        :param lambda_bound: weight for the bounding loss term
        :param device: device to run the computations on ('cuda', usually)

        TODO: kl divergence for different dofs
        """
        self.only_rank = True
        self.transition_time = transition_time
        self.lambda_flat_active = lambda_flat_active
        self.lambda_flat_inactive = lambda_flat_inactive
        
        self.device = torch.device(device)

    def __call__(self, predictions_batch, directions, onsets, print_components=False):
        return self.rank_loss(predictions_batch, directions, onsets, print_components)
    
    def rank_loss(self, predictions_batch, directions, onsets, print_components=False):
        """
        Compute only the rank loss + flatness loss component.
        
        :param self: Rank loss instance
        :param predictions_batch: predictions for the batch, shape [batch_size, D, N], or [N, D] if batch_size=1
        :param directions: directions for each dof for the particular trial, shape [batch_size, D], 1 for positive (flex), -1 for negative (extend), or D if batch_size=1
        :param onsets: onset indices for each dof in the batch, shape [batch_size, D], or D if batch_size=1
        """
        if predictions_batch.ndim != 3:
            predictions_batch = predictions_batch.unsqueeze(0).permute(0, 2, 1)  # shape [1, D, N]
            onsets = onsets.unsqueeze(0)
            directions = directions.unsqueeze(0)

        batch_size, D, N = predictions_batch.shape

        rank_loss = torch.zeros((), device=predictions_batch.device)
        flat_loss_active = torch.zeros((), device=predictions_batch.device)
        flat_loss_inactive = torch.zeros((), device=predictions_batch.device)

        valid_pairs = 0
        for i in range(batch_size):
            for dof in range(D):
                onset = onsets[i, dof]

                if torch.isnan(onset):
                    continue # skip if no onset

                cur_predictions = predictions_batch[i, dof, :]
                direction = directions[i, dof]

                if onset < 0 or onset >= N:
                    # DoF is inactive
                    flat_loss_inactive += torch.var(cur_predictions[onset:onset+self.transition_time])  # encourage flat predictions during inactive period
                    continue
                
                preonset = cur_predictions[:onset]
                postonset = cur_predictions[onset:]

                if preonset.numel() == 0 or postonset.numel() == 0:
                    continue

                # Compute pairwise differences via broadcasting
                # shape: (k, N-k)
                diffs = postonset[None, :] - preonset[:, None]

                if direction == -1:
                    diffs = -diffs

                # Logistic / RankNet loss
                rank_loss += torch.mean(torch.nn.functional.softplus(-diffs))

                valid_pairs += 1
                
                # Flatness loss after onset
                if postonset.numel() > self.transition_time:
                    flat_loss_active += torch.var(postonset[self.transition_time:])  # encourage flat predictions after onset

        if valid_pairs == 0:
            warnings.warn("No valid pairs for rank loss calculation. Returning None loss.")
            return None
        
        rank_loss = rank_loss / (batch_size * D)
        flat_loss_active = flat_loss_active / (batch_size * D)
        flat_loss_inactive = flat_loss_inactive / (batch_size * D)

        if print_components:
            print(f"Rank Loss: {rank_loss.item():.4f}, Flat Loss Active: {flat_loss_active.item():.4f}, Flat Loss Inactive: {flat_loss_inactive.item():.4f}")

        return rank_loss + self.lambda_flat_active * flat_loss_active + self.lambda_flat_inactive * flat_loss_inactive
