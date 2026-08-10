import torch
import numpy as np
import warnings

class RankLoss:
    def __init__(self, transition_time = 5, lambda_flat_active = 0, lambda_flat_inactive = 1000, device='cuda'):
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

    def __call__(self, predictions_batch, y, directions, onsets, print_components=False, fullhist=False):
        if fullhist:
            return self.rank_loss_fullhist(predictions_batch, y, directions, onsets, print_components)
        else:
            return self.rank_loss(predictions_batch, y, directions, onsets, print_components)

    def rank_loss_fullhist(self, predictions_batch, y, directions, onsets, print_components=False):
        """
        Compute the rank loss + flatness loss component, using the full history of predictions.

        :param self: Rank loss instance
        :param predictions_batch: predictions for the run, shape [N, D]
        :param directions: directions of current trial per bin, shape [N, D], 1 for positive (flex), -1 for negative (extend) --> augmented by TrialToBins blcok
        :param onsets: onset of current trial per bin, shape [N, D] --> augmented by TrialToBins block
        """
        if predictions_batch.ndim == 1:
            predictions_batch = predictions_batch.unsqueeze(1)
            y = y.unsqueeze(1)
            directions = directions.unsqueeze(1)
            onsets = onsets.unsqueeze(1)

        rank_loss = torch.zeros((), device=predictions_batch.device)
        flat_loss = torch.zeros((), device=predictions_batch.device)

        directions_diff = torch.diff(directions, axis=0)
        onset_idx = torch.where(torch.any(directions_diff != 0, axis=1))[0] + 1 # this has extra onsets because of trial transitions, but algorithm ignores them because the directions are 0 for those bins
        onsets = onsets[onset_idx, :]
        directions = directions[onset_idx, :]

        D = predictions_batch.shape[1]
        
        for dof in range(D):
            n_flat = 0
            valid_onsets = onsets[:, dof] != 0  # filter out zero onsets
            dof_onsets = onsets[valid_onsets, dof]
            dof_directions = directions[valid_onsets, dof] # filter out corresponding directions

            prev_onsets = dof_onsets[:-2] # length ntrials-2
            next_onsets = dof_onsets[2:] # length ntrials-2
            dof_onsets = dof_onsets[1:-1] # length ntrials-2
            dof_directions = dof_directions[1:-1] # length ntrials-2

            if len(dof_onsets) == 0:
                D -= 1
                continue  # skip if no valid onsets for this dof, happens with small dataset

            dof_rank_loss = torch.zeros((), device=predictions_batch.device)
            dof_flat_loss = torch.zeros((), device=predictions_batch.device)

            for prev_onset, onset, next_onset, direction in zip(prev_onsets, dof_onsets, next_onsets, dof_directions):
                pre_onset = predictions_batch[prev_onset+self.transition_time:onset, dof]
                post_onset = predictions_batch[onset+self.transition_time:next_onset, dof]

                diffs = post_onset[None, :] - pre_onset[:, None]

                if direction == -1:
                    diffs = -diffs

                dof_rank_loss += torch.mean(torch.nn.functional.softplus(-diffs))
                dof_flat_loss += torch.var(post_onset) * (next_onset - onset - self.transition_time)  # encourage flat predictions after onset
                n_flat += (next_onset - onset - self.transition_time)

            rank_loss += dof_rank_loss / dof_onsets.shape[0] # normalize by number of trials 
            flat_loss += dof_flat_loss / n_flat # normalize by number of bins contributing to flatness loss

        if print_components:
            print(f"Rank Loss: {rank_loss.item():.4f}, Flat Loss: {flat_loss.item():.4f}")

        rank_loss = rank_loss / D

        if torch.isnan(rank_loss):
            rank_loss = torch.tensor(0.0, device=rank_loss.device) # not enough data to compute loss, return zero

        return rank_loss + self.lambda_flat_active * flat_loss
                
    def rank_loss(self, predictions_batch, y, directions, onsets, print_components=False):
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

                if direction == 0:
                    # DoF is inactive
                    flat_loss_inactive += torch.var(cur_predictions)  # encourage flat predictions during inactive period
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

class RankMSEDOFLoss:
    def __init__(self, dof_loss = [1,0], transition_time = 5, lambda_flat_active = 0, lambda_flat_inactive = 1000, device='cuda'):
        """
        
        :param self: RankLoss instance
        :param dof_loss: how much to weight rank vs. MSE for each degree of freedom
        :param transition_time: number of time steps after onset to apply flatness loss
        :param lambda_bound: weight for the bounding loss term
        :param device: device to run the computations on ('cuda', usually)

        TODO: kl divergence for different dofs
        """
        self.dof_loss = dof_loss
        self.rank_loss_class = RankLoss(transition_time, lambda_flat_active, lambda_flat_inactive, device)
        self.mse_loss = torch.nn.MSELoss()
        
        self.device = torch.device(device)

    def __call__(self, predictions_batch, y, directions, onsets, print_components=False, fullhist=False):

        return self.rankmsedofloss(predictions_batch, y, directions, onsets, print_components)

    def rankmsedofloss(self, predictions_batch, y, directions, onsets, print_components=False):

        D = y.shape[1]
        if D != len(self.dof_loss):
            raise ValueError("Number of degrees of freedom in y must match the length of dof_loss")

        total_loss = 0
        for i, lambda_rank in enumerate(self.dof_loss):
            rank_loss_dof = self.rank_loss_class(predictions_batch[:, i], y[:, i], directions[:, i], onsets[:, i], print_components, fullhist=True)
            mse_loss_dof = self.mse_loss(y[:, i], predictions_batch[:, i])
            total_loss += lambda_rank * rank_loss_dof + (1 - lambda_rank) * mse_loss_dof

        return total_loss
