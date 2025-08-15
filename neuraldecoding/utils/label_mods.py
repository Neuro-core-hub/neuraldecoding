import torch
import numpy as np
from typing import List, Tuple, Union
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import warnings
from neuraldecoding.utils.loss_functions import dilate_loss

def apply_modifications(nicknames, kinematics, interpipe, param_dict):
    trial_filt = interpipe['trial_filt'][interpipe['train_mask']]
    targets_filt = interpipe['targets_filt'][interpipe['train_mask']]

    for name in nicknames:
        current_params = param_dict[name]
        mod = current_params['mod_type']
        if mod == 'shift_bins':
            kinematics = shift_kinematics(kinematics, current_params['shift'])
        elif mod == 'shift_by_trial':
            kinematics = shift_kinematics_by_trial(kinematics, trial_filt, current_params['shift_range'], current_params['individuate_dofs'])
        elif mod == 'warp_by_trial':
            kinematics = warp_kinematics_by_trial(kinematics, trial_filt, current_params['warp_factor'], current_params['hold_time'])
        elif mod == 'random_warp':
            kinematics = random_warp(kinematics, trial_filt, current_params['hold_time'], current_params['individuate_dofs'])
        elif mod == 'sigmoid_replacement':
            kinematics = replace_with_sigmoid(kinematics, trial_filt, targets_filt, current_params['sigmoid_k'], current_params['center'])
        elif mod == 'bias_endpoints':
            kinematics = bias_endpoints(kinematics, trial_filt, current_params['bias_range'], current_params['individuate_dofs'])
        else:
            warnings.warn(f'Modification {mod} is not an option. Skipping...')
    
    return kinematics


def shift_kinematics(kinematics: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shift kinematics data forward or backward in time.

    Args:
        kinematics: Tensor of shape (T, N) where N is the number of kinematic variables
                   (first half positions, second half velocities)
        trial_indices: Array of shape (T,) containing trial indices for each time point
        shift: Number of time steps to shift (positive for forward, negative for backward)

    Returns:
        Shifted kinematics tensor of the same shape
    """
    if not isinstance(kinematics, torch.Tensor):
        kinematics = torch.tensor(kinematics, dtype=torch.float32)

    if shift == 0:
        return kinematics.clone()

    T, N = kinematics.shape
    shifted = torch.zeros_like(kinematics)

    if shift > 0:
        # Shift forward
        shifted[shift:] = kinematics[:-shift]
        # For beginning points, repeat the first valid value
        for i in range(shift):
            shifted[i] = kinematics[0]
    else:
        # Shift backward
        shifted[:shift] = kinematics[-shift:]
        # For end points, repeat the last valid value
        for i in range(-shift):
            shifted[T - i - 1] = kinematics[T - 1]

    return shifted

def shift_kinematics_by_trial(
    kinematics: torch.Tensor, trial_indices: np.ndarray, shift_range: Tuple[int, int], individuate_dofs: bool = False, 
) -> torch.Tensor:
    """
    Shift kinematics data by different amounts for each trial.

    Args:
        kinematics: Tensor of shape (T, N) where N is the number of kinematic variables
        trial_indices: Array of shape (T,) containing trial indices for each time point
        shift_range: Tuple (min_shift, max_shift) defining the range of shifts to apply

    Returns:
        Shifted kinematics tensor of the same shape
    """
    if not isinstance(kinematics, torch.Tensor):
        kinematics = torch.tensor(kinematics, dtype=torch.float32)
        
    min_shift, max_shift = shift_range
    unique_trials = np.unique(trial_indices)
    shifted = kinematics.clone()

    N = kinematics.shape[1]
    pos_dim = N // 2

    for trial in unique_trials:
        # Get mask for this trial
        trial_mask = trial_indices == trial
        trial_indices_positions = np.where(trial_mask)[0]

        if len(trial_indices_positions) == 0:
            continue

        start_idx = trial_indices_positions[0]
        end_idx = trial_indices_positions[-1] + 1
        trial_length = end_idx - start_idx

        trial_data = kinematics[trial_mask]
        shifted_trial_data = torch.zeros_like(trial_data)
        
        if individuate_dofs:
            trial_shift = np.random.randint(min_shift, max_shift + 1, size=pos_dim)
            for idx, shift in enumerate(trial_shift):
                dims_to_change = np.array([idx, idx+pos_dim])
                if shift > 0:
                    # Shift forward
                    if shift >= trial_length:
                        # If shift is larger than trial, just use first value
                        shifted_trial_data[:,dims_to_change] = trial_data[0,dims_to_change].unsqueeze(0).repeat(trial_length, 1)
                    else:
                        shifted_trial_data[shift:,dims_to_change] = trial_data[:-shift,dims_to_change]
                        # For beginning points, repeat the first valid value
                        for i in range(shift):
                            shifted_trial_data[i,dims_to_change] = trial_data[0,dims_to_change]
                else:
                    # Shift backward
                    abs_shift = abs(shift)
                    if abs_shift >= trial_length:
                        # If shift is larger than trial, just use last value
                        shifted_trial_data[:,dims_to_change] = trial_data[-1,dims_to_change].unsqueeze(0).repeat(trial_length, 1)
                    else:
                        shifted_trial_data[: trial_length - abs_shift, dims_to_change] = trial_data[abs_shift:,dims_to_change]
                        # For end points, repeat the last valid value
                        for i in range(abs_shift):
                            shifted_trial_data[trial_length - i - 1,dims_to_change] = trial_data[-1,dims_to_change]
        else:
            # Generate random shift for this trial
            trial_shift = np.random.randint(min_shift, max_shift + 1)

            if trial_shift > 0:
                # Shift forward
                if trial_shift >= trial_length:
                    # If shift is larger than trial, just use first value
                    shifted_trial_data = trial_data[0].unsqueeze(0).repeat(trial_length, 1)
                else:
                    shifted_trial_data[trial_shift:] = trial_data[:-trial_shift]
                    # For beginning points, repeat the first valid value
                    for i in range(trial_shift):
                        shifted_trial_data[i] = trial_data[0]
            else:
                # Shift backward
                abs_shift = abs(trial_shift)
                if abs_shift >= trial_length:
                    # If shift is larger than trial, just use last value
                    shifted_trial_data = trial_data[-1].unsqueeze(0).repeat(trial_length, 1)
                else:
                    shifted_trial_data[: trial_length - abs_shift] = trial_data[abs_shift:]
                    # For end points, repeat the last valid value
                    for i in range(abs_shift):
                        shifted_trial_data[trial_length - i - 1] = trial_data[-1]

        # Update the output tensor with shifted trial data
        shifted[trial_mask] = shifted_trial_data

        """
        loss_index = dilate_loss(shifted[trial_mask, 0].unsqueeze(0).unsqueeze(2), kinematics[trial_mask, 0].unsqueeze(0).unsqueeze(2), 1, 0.001, device=kinematics.device)
        loss_mrs = dilate_loss(shifted[trial_mask, 1].unsqueeze(0).unsqueeze(2), kinematics[trial_mask, 1].unsqueeze(0).unsqueeze(2), 1, 0.001, device=kinematics.device)
        print(loss_index)
        print(loss_mrs)
        print(trial_shift)
        plt.plot(np.arange(trial_length), shifted[trial_mask, 0].cpu().numpy(), color='b', label='shifted index pos')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 0].cpu().numpy(), color='b', label='true index pos', linestyle='--')
        plt.plot(np.arange(trial_length), shifted[trial_mask, 1].cpu().numpy(), color='r', label='shifted mrs pos')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 1].cpu().numpy(), color='r', label='true mrs pos', linestyle='--')
        plt.xlabel('bins')
        plt.ylabel('Position (a.u.)')
        plt.legend()
        plt.show()

        plt.plot(np.arange(trial_length), shifted[trial_mask, 2].cpu().numpy(), color='g', label='shifted index vel')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 2].cpu().numpy(), color='g', label='true index vel', linestyle='--')
        plt.plot(np.arange(trial_length), shifted[trial_mask, 3].cpu().numpy(), color='y', label='shifted mrs vel')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 3].cpu().numpy(), color='y', label='true mrs vel', linestyle='--')
        plt.xlabel('bins')
        plt.ylabel('Velocity (a.u.)')
        plt.legend()
        plt.show()
        """

    return shifted

def random_warp(
    kinematics: torch.Tensor,
    trial_indices: np.ndarray,
    hold_time: int,
    individuate_dofs: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    Warp the kinematics by interpolating the trajectory and randomly resampling the interpolated points

    Args:
        kinematics: Tensor of shape (T, N) where N is the number of kinematic variables
        trial_indices: Array of shape (T,) containing trial indices for each time point
        hold_time: Number of time points at the end of each trial to consider as "hold time"
                  This is the period that will be adjusted when warping

    Returns:
        Warped kinematics tensor of the same shape
    """
    if not isinstance(kinematics, torch.Tensor):
        kinematics = torch.tensor(kinematics, dtype=torch.float32)
    
    unique_trials = np.unique(trial_indices)
    warped = kinematics.clone()

    # Get the number of dimensions and separate position and velocity
    N = kinematics.shape[1]
    pos_dim = N // 2

    for trial in unique_trials:
        # Get mask for this trial
        trial_mask = trial_indices == trial
        trial_indices_positions = np.where(trial_mask)[0]

        if len(trial_indices_positions) <= hold_time + 1:
            continue  # Skip trials that are too short

        trial_data = kinematics[trial_mask]
        trial_length = len(trial_data)

        warped_trial_data = trial_data.clone()

        x = np.arange(trial_length - hold_time)

        xresamp = np.random.uniform(0, trial_length - hold_time, size = trial_length - hold_time)
        xresamp.sort()
        for dim in range(pos_dim):
            # overwrites the xresamp for each dof
            if individuate_dofs:
                xresamp = np.random.uniform(0, trial_length - hold_time, size = trial_length - hold_time)
                xresamp.sort()

            # Interpolate the position data for this dimension
            y = trial_data[:trial_length - hold_time, dim].cpu().numpy()
            y_interp = torch.Tensor(np.interp(xresamp, x, y))


            # Fill the warped trial data with the interpolated values
            warped_trial_data[:trial_length - hold_time, dim] = y_interp
            warped_trial_data[trial_length - hold_time:trial_length, dim] = trial_data[trial_length - hold_time:, dim]
        
        warped[trial_mask] = warped_trial_data.to(device=kinematics.device)
        
        warped[1:, pos_dim:2*pos_dim] = warped[1:, :pos_dim] - warped[:-1, :pos_dim]

        """
        loss_index = dilate_loss(warped[trial_mask, 0].unsqueeze(0).unsqueeze(2), kinematics[trial_mask, 0].unsqueeze(0).unsqueeze(2), 1, 0.001, device=kinematics.device)
        loss_mrs = dilate_loss(warped[trial_mask, 1].unsqueeze(0).unsqueeze(2), kinematics[trial_mask, 1].unsqueeze(0).unsqueeze(2), 1, 0.001, device=kinematics.device)
        print(loss_index)
        print(loss_mrs)
        plt.plot(np.arange(trial_length), warped[trial_mask, 0].cpu().numpy(), color='b', label='warped index pos')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 0].cpu().numpy(), color='b', label='true index pos', linestyle='--')
        plt.plot(np.arange(trial_length), warped[trial_mask, 1].cpu().numpy(), color='r', label='warped mrs pos')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 1].cpu().numpy(), color='r', label='true mrs pos', linestyle='--')
        plt.xlabel('bins')
        plt.ylabel('Position (a.u.)')
        plt.legend()
        plt.show()

        plt.plot(np.arange(trial_length), warped[trial_mask, 2].cpu().numpy(), color='g', label='warped index vel')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 2].cpu().numpy(), color='g', label='true index vel', linestyle='--')
        plt.plot(np.arange(trial_length), warped[trial_mask, 3].cpu().numpy(), color='y', label='warped mrs vel')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 3].cpu().numpy(), color='y', label='true mrs vel', linestyle='--')
        plt.xlabel('bins')
        plt.ylabel('Velocity (a.u.)')
        plt.legend()
        plt.show()
        """
        
    return warped
        
def warp_kinematics_by_trial(
    kinematics: torch.Tensor,
    trial_indices: np.ndarray,
    warp_factor: float,
    hold_time: int,
) -> Tuple[torch.Tensor, float]:
    """
    Warp the kinematics data within each trial by making it faster or slower.

    Args:
        kinematics: Tensor of shape (T, N) where N is the number of kinematic variables
        trial_indices: Array of shape (T,) containing trial indices for each time point
        warp_factor: Float controlling the speed of the trajectory
                    Values < 1: slower (stretches the trajectory)
                    Values = 1: no change
                    Values > 1: faster (compresses the trajectory)
        hold_time: Number of time points at the end of each trial to consider as "hold time"
                  This is the period that will be adjusted when warping

    Returns:
        Warped kinematics tensor of the same shape
    """
    if not isinstance(kinematics, torch.Tensor):
        kinematics = torch.tensor(kinematics, dtype=torch.float32)

    unique_trials = np.unique(trial_indices)
    warped = kinematics.clone()

    # Get the number of dimensions and separate position and velocity
    N = kinematics.shape[1]
    pos_dim = N // 2  # First half is position

    actual_warp_factors = []
    for trial in unique_trials:
        # Get mask for this trial
        trial_mask = trial_indices == trial
        trial_indices_positions = np.where(trial_mask)[0]

        if len(trial_indices_positions) <= hold_time + 1:
            continue  # Skip trials that are too short

        trial_data = kinematics[trial_mask]
        trial_length = len(trial_data)

        # Determine the movement portion (excluding hold time)
        movement_length = trial_length - hold_time

        if movement_length <= 1:
            actual_warp_factors.append(1.0)
            continue  # Skip if there's not enough movement data

        # Get the final hold position (average of hold time points for stability)
        hold_position = trial_data[-hold_time:, :pos_dim].mean(dim=0)

        if warp_factor == 1.0:
            # No warping needed
            actual_warp_factors.append(1.0)
            continue

        elif warp_factor > 1.0:
            # Faster: compress the movement portion and extend the hold time
            # Calculate how many points to compress the movement into
            compressed_length = int(movement_length / warp_factor)
            if compressed_length < 1:
                compressed_length = 1

            # Interpolate the movement portion to the compressed length
            movement_data = trial_data[:movement_length, :pos_dim]
            compressed_indices = np.linspace(0, movement_length - 1, compressed_length)

            # Create the warped trial data (only for positions)
            warped_trial_data = torch.zeros(
                (trial_length, pos_dim), dtype=kinematics.dtype
            )

            # Interpolate each position dimension
            for i in range(pos_dim):
                # Compress the movement portion
                compressed_movement = torch.tensor(
                    np.interp(
                        compressed_indices,
                        np.arange(movement_length),
                        movement_data[:, i].cpu().numpy(),
                    ),
                    dtype=kinematics.dtype,
                )

                # Fill in the compressed movement
                warped_trial_data[:compressed_length, i] = compressed_movement

                # Fill the rest with the hold position plus some noise
                noise = (
                    torch.randn(
                        trial_length - compressed_length,
                        dtype=kinematics.dtype,
                        device=kinematics.device,
                    )
                    * 0.005
                )
                warped_trial_data[compressed_length:, i] = hold_position[i] + noise
                actual_warp_factors.append(movement_length / compressed_length)

        else:  # warp_factor < 1.0
            # Slower: stretch the movement portion and reduce the hold time
            # Calculate how many points to stretch the movement into
            stretched_length = int(movement_length / warp_factor)

            # Make sure we don't exceed the trial length
            if stretched_length > trial_length:
                stretched_length = trial_length

            # Interpolate the movement portion to the stretched length
            movement_data = trial_data[:movement_length, :pos_dim]
            stretched_indices = np.linspace(0, movement_length - 1, stretched_length)

            # Create the warped trial data (only for positions)
            warped_trial_data = torch.zeros(
                (trial_length, pos_dim), dtype=kinematics.dtype
            )
            actual_warp_factors.append(movement_length / stretched_length)

            # Interpolate each position dimension
            for i in range(pos_dim):
                # Stretch the movement portion
                stretched_movement = torch.tensor(
                    np.interp(
                        stretched_indices,
                        np.arange(movement_length),
                        movement_data[:, i].cpu().numpy(),
                    ),
                    dtype=kinematics.dtype,
                )

                # Fill in as much of the stretched movement as fits
                max_idx = min(stretched_length, trial_length)
                warped_trial_data[:max_idx, i] = stretched_movement[:max_idx]

                # If there's room left, fill with the hold position
                if max_idx < trial_length:
                    noise = (
                        torch.randn(
                            trial_length - max_idx,
                            dtype=kinematics.dtype,
                            device=kinematics.device,
                        )
                        * 0.005
                    )
                    warped_trial_data[max_idx:, i] = hold_position[i] + noise

        # Update the position part of the output tensor with warped trial data
        warped[trial_mask, :pos_dim] = warped_trial_data.to(device=kinematics.device)

        # Compute velocities from the warped positions
        # First point velocity is set to zero
        warped_velocities = torch.zeros((trial_length, pos_dim), dtype=kinematics.dtype)

        # Calculate velocities as position differences
        if trial_length > 1:
            warped_velocities[1:] = warped_trial_data[1:] - warped_trial_data[:-1]

        # Update the velocity part of the output tensor
        warped[trial_mask, pos_dim:] = warped_velocities.to(device=kinematics.device)
        """
        loss_index = dilate_loss(warped[trial_mask, 0].unsqueeze(0).unsqueeze(2), kinematics[trial_mask, 0].unsqueeze(0).unsqueeze(2), 1, 0.001, device=kinematics.device)
        loss_mrs = dilate_loss(warped[trial_mask, 1].unsqueeze(0).unsqueeze(2), kinematics[trial_mask, 1].unsqueeze(0).unsqueeze(2), 1, 0.001, device=kinematics.device)
        print(loss_index)
        print(loss_mrs)
        plt.plot(np.arange(trial_length), warped[trial_mask, 0].cpu().numpy(), color='b', label='warped index pos')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 0].cpu().numpy(), color='b', label='true index pos', linestyle='--')
        plt.plot(np.arange(trial_length), warped[trial_mask, 1].cpu().numpy(), color='r', label='warped mrs pos')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 1].cpu().numpy(), color='r', label='true mrs pos', linestyle='--')
        plt.xlabel('bins')
        plt.ylabel('Position (a.u.)')
        plt.legend()
        plt.show()

        plt.plot(np.arange(trial_length), warped[trial_mask, 2].cpu().numpy(), color='g', label='warped index vel')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 2].cpu().numpy(), color='g', label='true index vel', linestyle='--')
        plt.plot(np.arange(trial_length), warped[trial_mask, 3].cpu().numpy(), color='y', label='warped mrs vel')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 3].cpu().numpy(), color='y', label='true mrs vel', linestyle='--')
        plt.xlabel('bins')
        plt.ylabel('Velocity (a.u.)')
        plt.legend()
        plt.show()
        """

    # Compute average warp factor actually applied
    print(f"Average warp factor: {np.mean(actual_warp_factors)}")
    return warped, np.mean(actual_warp_factors)

def replace_with_sigmoid(
    kinematics: torch.Tensor,
    trial_indices: np.ndarray,
    targets: np.ndarray,
    sigmoid_k: float = 1.0,
    center: int = 0.5
) -> torch.Tensor:
    
    def sigmoid(x, x0=0, k=1, y_start=0, y_end=1):
        return torch.tensor(y_start + (y_end - y_start) / (1 + np.exp(-k * (x - x0)))).to(dtype=torch.float32)
    
    if not isinstance(kinematics, torch.Tensor):
        kinematics = torch.tensor(kinematics, dtype=torch.float32)
    
    unique_trials = np.unique(trial_indices)
    sigmoid_data = kinematics.clone()

    # Get the number of dimensions and separate position and velocity
    N = kinematics.shape[1]
    pos_dim = N // 2
    prev_target = None

    for trial in unique_trials:
        # Get mask for this trial
        trial_mask = trial_indices == trial

        trial_data = kinematics[trial_mask]
        trial_length = len(trial_data)

        trial_target = targets[trial_mask][0]
        if prev_target is None:
            prev_target = trial_target
            continue

        sigmoid_trial_data = trial_data.clone()

        for dim in range(pos_dim):
            y_start = prev_target[dim]
            y_end = trial_target[dim]
            x = np.arange(trial_length)
            x0 = trial_length * center
            y_new = sigmoid(x, x0=x0, k=sigmoid_k, y_start=y_start, y_end=y_end)

            # Fill the warped trial data with the interpolated values
            sigmoid_data[trial_mask, dim] = y_new
            
            if trial_length > 1:
                sigmoid_trial_data[1:, dim+pos_dim] = y_new[1:] - y_new[:-1]
            
            sigmoid_data[trial_mask, dim+pos_dim] = sigmoid_trial_data[:, dim+pos_dim]
        
        prev_target = trial_target

        """
        loss_index = dilate_loss(sigmoid_data[trial_mask, 0].unsqueeze(0).unsqueeze(2), kinematics[trial_mask, 0].unsqueeze(0).unsqueeze(2), 1, 0.001, device=kinematics.device)
        loss_mrs = dilate_loss(sigmoid_data[trial_mask, 1].unsqueeze(0).unsqueeze(2), kinematics[trial_mask, 1].unsqueeze(0).unsqueeze(2), 1, 0.001, device=kinematics.device)
        print(loss_index)
        print(loss_mrs)
        plt.plot(np.arange(trial_length), sigmoid_data[trial_mask, 0].cpu().numpy(), color='b', label='warped index pos')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 0].cpu().numpy(), color='b', label='true index pos', linestyle='--')
        plt.plot(np.arange(trial_length), sigmoid_data[trial_mask, 1].cpu().numpy(), color='r', label='warped mrs pos')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 1].cpu().numpy(), color='r', label='true mrs pos', linestyle='--')
        plt.xlabel('bins')
        plt.ylabel('Position (a.u.)')
        plt.legend()
        plt.show()

        plt.plot(np.arange(trial_length), sigmoid_data[trial_mask, 2].cpu().numpy(), color='g', label='warped index vel')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 2].cpu().numpy(), color='g', label='true index vel', linestyle='--')
        plt.plot(np.arange(trial_length), sigmoid_data[trial_mask, 3].cpu().numpy(), color='y', label='warped mrs vel')
        plt.plot(np.arange(trial_length), kinematics[trial_mask, 3].cpu().numpy(), color='y', label='true mrs vel', linestyle='--')
        plt.xlabel('bins')
        plt.ylabel('Velocity (a.u.)')
        plt.legend()
        plt.show()
        """
        
    return sigmoid_data

def bias_endpoints(kinematics: torch.Tensor,
    trial_indices: np.ndarray,
    bias_range: Union[float, Tuple[float, float]],
    individuate_dofs: bool = False
) -> torch.Tensor:
    """    Bias the endpoints of each trial by adding a random bias to the first and last position values.
    Args:
        kinematics: Tensor of shape (T, N) where N is the number of kinematic variables
                   (first half positions, second half velocities)
        trial_indices: Array of shape (T,) containing trial indices for each time point
        bias_range: Either a float (symmetric range) or a tuple (min_bias, max_bias)
                   defining the range of endpoint biases to apply
    Returns:
        Modified kinematics tensor with biased endpoints
    """
    if not isinstance(kinematics, torch.Tensor):
        kinematics = torch.tensor(kinematics, dtype=torch.float32)

    if isinstance(bias_range, (int, float)):
        bias_max = bias_range
        bias_min = -bias_range
    else:
        bias_min, bias_max = bias_range[0], bias_range[1]
    
    unique_trials = np.unique(trial_indices)
    biased = kinematics.clone()

    # Get the number of dimensions and separate position and velocity
    N = kinematics.shape[1]
    pos_dim = N // 2
    prev_bias = np.zeros(pos_dim, dtype=np.float32)
    for trial in unique_trials:
        # Get mask for this trial
        trial_mask = trial_indices == trial
        if individuate_dofs:
            current_bias = np.random.uniform(bias_min, bias_max, size=pos_dim)
        else:
            current_bias = np.random.uniform(bias_min, bias_max)
            current_bias = np.repeat(current_bias, pos_dim)

        trial_data = kinematics[trial_mask]
        new_trial_data = trial_data.clone()

        first_pos = trial_data[0, :pos_dim]
        last_pos = trial_data[-1, :pos_dim]

        firstplusbias = first_pos + prev_bias
        lastplusbias = last_pos + current_bias
        scale = (lastplusbias - firstplusbias) / (last_pos - first_pos)

        new_trial_data[:, :pos_dim] = (trial_data[:, :pos_dim] - first_pos) * scale + firstplusbias

        biased[trial_mask, :pos_dim] = new_trial_data[:, :pos_dim]

        prev_bias = current_bias

    for dim in range(pos_dim):
        # Get the positions for this dimension
        pos = biased[:, dim].numpy()

        # Compute gradient for all points using numpy's gradient function
        # This automatically handles the endpoints appropriately
        vel = np.diff(pos, axis=0, append=0)

        # Update the velocity in the output tensor
        biased[:, pos_dim + dim] = torch.tensor(vel, dtype=kinematics.dtype)

    return biased