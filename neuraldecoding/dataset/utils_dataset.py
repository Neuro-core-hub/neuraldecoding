import os
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def calc_bins_sbp(sbp_data, sample_width, bins_start_digit, bins_stop_digit):
    """
    Calculate the bins Spiking Band Power (SBP) for binned SBP data

    Inputs:
        sbp_data: np.array (n_times, n_channels)
            SBP data
        sample_width: np.array (n_times, 1)
            Sample width data
        bins_start_digit: np.array
            Bins start digitized
        bins_stop_digit: np.array
            Bins stop digitized

    Outputs:
        bins_sbp: np.array
            Bins SBP data
    """
    bins_sbp = np.zeros((len(bins_start_digit), sbp_data.shape[1]))

    for i in range(len(bins_start_digit)):
        start = bins_start_digit[i]
        stop = bins_stop_digit[i]

        indices_i = np.linspace(start, stop, abs(stop - start) + 1, dtype=int)
        bins_sbp[i] = np.sum(sbp_data[indices_i, :], axis=0) / np.sum(
            sample_width[indices_i]
        )

    return bins_sbp

def calc_bins_fingers_kinematics(fingers_pos_data, bins_start, bins_stop, num_fingers):
    """
    Calculate the bins fingers kinematics for binned fingers kinematics data

    Inputs:
        fingers_pos_data: np.array (n_times, n_fingers)
            Fingers position data
        bins_start: np.array
            Bins start
        bins_stop: np.array
            Bins stop
        num_fingers: int
            Number of fingers

    Outputs:
        bins_kinematics: np.array (n_bins, n_fingers + 2 * n_fingers)
            Bins fingers kinematics data
    """

    num_samples = len(bins_start)
    bins_kinematics = np.zeros((num_samples, num_fingers))

    # For each index i, get the features between bins_start[i] and bins_stop[i], and take their mean
    for i in range(num_samples):
        bins_kinematics[i] = np.mean(
            fingers_pos_data[bins_start[i] : bins_stop[i]], axis=0
        )

    # computing velocity
    kinematics_first_diff = np.diff(bins_kinematics, n=1, axis=0)
    kinematics_second_diff = np.diff(bins_kinematics, n=2, axis=0)

    # Rows of zeros for padding
    one_row_zeros = np.zeros([1, num_fingers])
    two_row_zeros = np.zeros([2, num_fingers])

    # Construct first and second difference matrices
    first_diff_matrix = np.concatenate((kinematics_first_diff, one_row_zeros), axis=0)
    second_diff_matrix = np.concatenate((kinematics_second_diff, two_row_zeros), axis=0)

    diff_matrix = np.concatenate((first_diff_matrix, second_diff_matrix), axis=1)

    # Concatenate diffMatrix to the right side of run_features
    bins_kinematics = np.concatenate((bins_kinematics, diff_matrix), axis=1)

    return bins_kinematics