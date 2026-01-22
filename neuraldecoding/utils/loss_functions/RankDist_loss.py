import torch
import numpy as np

class RankDistLoss:
    def __init__(self, vel_preds_path, binmax=9, binmin=-9, nbins=99, min_bound=-0.5, max_bound = 0.5, transition_time = 5, lambda_kl = 0, lambda_bound = 0, lambda_flat = 40, device='cuda'):
        """
        
        :param self: RankDistLoss instance
        :param vel_preds_path: path to the predicted velocities used to compute the ground truth velocity distribution, saved as a .npy file, numpy array of shape [N, D]
        :param scaler_path: path to the scaler used for normalizing the velocities, saved as a .pkl file, sklearn StandardScaler object
        :param nbins: number of bins for the velocity histogram
        :param lambda_kl: weight for the kl divergence regularization term
        :param lambda_bound: weight for the bounding loss term
        :param device: device to run the computations on ('cuda', usually)

        TODO: kl divergence for different dofs
        """
        vel_preds = np.load(vel_preds_path)

        binedges = np.linspace(binmin, binmax, nbins+1)
        # bin centers
        bin_centers = 0.5 * (binedges[:-1] + binedges[1:])
        sigma = binedges[1] - binedges[0]

        # flatten velocities across time and DOFs
        vel_preds_flat = vel_preds.reshape(-1, 1)

        # soft histogram (numpy version)
        weights = np.exp(-0.5 * ((vel_preds_flat - bin_centers[None, :]) / sigma) ** 2)
        v_hist = weights.mean(axis=0)

        self.gt_dist = torch.tensor(
            v_hist / (np.sum(v_hist) + 1e-8),
            device=device,
            dtype=torch.float32
        )

        self.min_bound = min_bound
        self.max_bound = max_bound

        self.bincenters = torch.tensor(bin_centers, device=device, dtype=torch.float32)

        self.transition_time = transition_time

        self.lambda_kl = lambda_kl
        self.lambda_bound = lambda_bound
        self.lambda_flat = lambda_flat
        
        self.device = torch.device(device)

    def __call__(self, predictions_batch, predictions_full, directions, onsets):
        return self.rank_dist_loss(predictions_batch, predictions_full, directions, onsets)
    
    def rank_dist_loss(self, predictions_batch, predictions_full, directions, onsets, only_rank=False):
        """
        Data should be loaded with TrialHistory activated, and the a loss-compatible trainer should be used. Only position should be predicted.
        
        :param self: RankDistLoss instance
        :param predictions_batch: predictions for the batch, shape [batch_size, N, D]
        :param predictions_full: full predictions, shape [N, D], used for kl divergence regularization
        :param directions: directions for each dof for the particular trial, shape [batch_size, D], 1 for positive (flex), -1 for negative (extend)
        :param onsets: onset indices for each dof in the batch, shape [batch_size, D]
        """
        batch_size, D, N = predictions_batch.shape

        rank_loss = torch.zeros((), device=predictions_batch.device)
        flat_loss = torch.zeros((), device=predictions_batch.device)
        valid_pairs = 0
        for i in range(batch_size):
            for dof in range(D):
                onset = onsets[i, dof]

                if torch.isnan(onset):
                    continue # skip if no onset

                cur_predictions = predictions_batch[i, dof, :]
                direction = directions[i, dof]

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
                flat_loss += torch.var(postonset[self.transition_time:])  # encourage flat predictions after onset

        if valid_pairs == 0:
            return None
        
        rank_loss = rank_loss / (batch_size * D)
        flat_loss = flat_loss / (batch_size * D)

        pred_vels = predictions_full[1:] - predictions_full[:-1]  # (N-1, D)

        sigma = self.bincenters[1] - self.bincenters[0]

        x = pred_vels.reshape(-1, 1)          # (T*D, 1)
        centers = self.bincenters.view(1, -1)        # (1, B)

        weights = torch.exp(-0.5 * ((x - centers) / sigma) ** 2)
        hist = weights.mean(dim=0)

        pred_dist = hist / (hist.sum() + 1e-8)

        kl_loss = torch.sum(
            self.gt_dist * (torch.log(self.gt_dist + 1e-8) - torch.log(pred_dist + 1e-8))
        )

        # Bounding loss
        lower = torch.relu(self.min_bound - predictions_full)
        upper = torch.relu(predictions_full - self.max_bound)
        bound_loss = torch.mean(lower**2 + upper**2)/(batch_size * D)

        total_loss = rank_loss + self.lambda_kl * kl_loss + self.lambda_bound * bound_loss + self.lambda_flat * flat_loss

        return total_loss
    
    def rank_flat_loss_only(self, predictions_batch, directions, onsets):
        """
        Compute only the rank loss + flatness loss component.
        
        :param self: RankDistLoss instance
        :param predictions_batch: predictions for the batch, shape [batch_size, N, D]
        :param directions: directions for each dof for the particular trial, shape [batch_size, D], 1 for positive (flex), -1 for negative (extend)
        :param onsets: onset indices for each dof in the batch, shape [batch_size, D]
        """
        batch_size, D, N = predictions_batch.shape

        rank_loss = torch.zeros((), device=predictions_batch.device)
        flat_loss = torch.zeros((), device=predictions_batch.device)
        valid_pairs = 0
        for i in range(batch_size):
            for dof in range(D):
                onset = onsets[i, dof]

                if torch.isnan(onset):
                    continue # skip if no onset

                cur_predictions = predictions_batch[i, dof, :]
                direction = directions[i, dof]

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
                flat_loss += torch.var(postonset[self.transition_time:])  # encourage flat predictions after onset

        if valid_pairs == 0:
            return None, None
        
        rank_loss = rank_loss / (batch_size * D)
        flat_loss = flat_loss / (batch_size * D)

        return rank_loss, flat_loss

    
    def dist_loss_only(self, predictions_full):
        """
        Compute only the distribution matching loss component (KL divergence + bounding loss).
        
        :param self: RankDistLoss instance
        :param predictions_full: full predictions, shape [N, D], used for kl divergence regularization
        """
        N, D = predictions_full.shape

        pred_vels = predictions_full[1:] - predictions_full[:-1]  # (N-1, D)

        sigma = self.bincenters[1] - self.bincenters[0]

        x = pred_vels.reshape(-1, 1)          # (T*D, 1)
        centers = self.bincenters.view(1, -1)        # (1, B)

        weights = torch.exp(-0.5 * ((x - centers) / sigma) ** 2)
        hist = weights.mean(dim=0)

        pred_dist = hist / (hist.sum() + 1e-8)

        kl_loss = torch.sum(
            self.gt_dist * (torch.log(self.gt_dist + 1e-8) - torch.log(pred_dist + 1e-8))
        )

        # Bounding loss
        lower = torch.relu(self.min_bound - predictions_full)
        upper = torch.relu(predictions_full - self.max_bound)
        bound_loss = torch.mean(lower**2 + upper**2)/(N * D)

        return kl_loss, bound_loss

    