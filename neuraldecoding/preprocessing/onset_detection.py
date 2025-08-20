import numpy as np
from scipy.ndimage import uniform_filter1d


class MovementOnsetDetector:
    """
    A class for detecting movement onsets from EMG signals using energy-CFAR method.
    
    This class encapsulates the entire pipeline from raw EMG processing to onset detection
    with configurable parameters provided through a config dictionary.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the MovementOnsetDetector with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with the following optional keys:
                - fs (int): Sampling frequency in Hz. Default: 2000
                - rms_win_ms (int): RMS window length in ms. Default: 15
                - cfar_win_ms (int): CFAR window length in ms. Default: 786
                - k_high (float): High threshold multiplier for CFAR. Default: 1.78
                - k_low (float): Low threshold multiplier for CFAR. Default: 4.01
                - aggregate (str): Channel aggregation method ('rss', 'mean', or None). Default: 'rss'
                - do_refine_onset (bool): Whether to refine onset timing. Default: True
                - guard_frac (float): Guard fraction for onset refinement. Default: 0.10
                - deadzone_ms (float): Deadzone duration at trial start in ms. Default: 100
        """
        # Set default configuration
        default_config = {
            'fs': 1000,
            'rms_win_ms': 15,
            'cfar_win_ms': 786,
            # 'cfar_win_ms': 500,
            'k_high': 1.78,
            'k_low': 4.01,
            'aggregate': 'rss',
            'do_refine_onset': True,
            'guard_frac': 0.1,
            'deadzone_ms': 0
        }
        
        # Update with user-provided config
        self.config = {**default_config, **config}
        
    def _tkeo(self, x: np.ndarray) -> np.ndarray:
        """Teager‑Kaiser Energy Operator, per channel."""
        e = x**2
        e[1:-1] -= x[:-2] * x[2:]
        return e

    def _centered_rms(self, x: np.ndarray, win: int) -> np.ndarray:
        """Zero‑phase moving RMS with a centred window of length *win* samples."""
        mean_sq = uniform_filter1d(x**2, size=win, axis=0, mode="reflect", origin=0)
        return np.sqrt(np.maximum(mean_sq, 1e-12))

    def _centered_mean_var(self, x: np.ndarray, win: int):
        """Zero‑phase running mean & std‑dev (σ) for CFAR (constant false alarm rate)."""
        mu = uniform_filter1d(x, size=win, axis=0, mode="reflect", origin=0)
        var = uniform_filter1d(x**2, size=win, axis=0, mode="reflect", origin=0) - mu**2
        return mu, np.sqrt(np.maximum(var, 1e-12))

    def _refine_onset(self, env: np.ndarray, coarse_idx: int) -> int:
        """
        Walk backwards from 'coarse_idx' until envelope < guard_frac * peak.
        Returns refined index (>=0).
        """
        peak = env[coarse_idx]
        thr = self.config['guard_frac'] * peak
        t = coarse_idx
        while t > 0 and env[t] > thr:
            t -= 1
        return t + 1

    def _energy_cfar_onsets_offline(self, emg: np.ndarray, return_env: bool = False):
        """
        Energy‑CFAR (constant false alarm rate) onset detector (offline, non‑causal).
        Returns 'onsets' (sample indices) and, optionally, the envelope used.
        """
        # 1. Energy operator
        e = self._tkeo(emg)

        # 2. RMS envelope (zero‑phase)
        rms_win = int(self.config['rms_win_ms'] * self.config['fs'] / 1000)
        env = self._centered_rms(e, rms_win)

        # 3. Aggregate channels if requested
        if self.config['aggregate'] == "rss":
            env_use = np.sqrt(np.sum(env**2, axis=1, keepdims=True))
        elif self.config['aggregate'] == "mean":
            env_use = np.mean(env, axis=1, keepdims=True)
        else:  # per‑channel
            env_use = env

        # 4. CFAR thresholds (zero‑phase)
        cfar_win = int(self.config['cfar_win_ms'] * self.config['fs'] / 1000)
        mu, sigma = self._centered_mean_var(env_use, cfar_win)
        th_hi = mu + self.config['k_high'] * sigma
        th_lo = mu + self.config['k_low'] * sigma

        # 5. Calculate deadzone duration in samples
        deadzone_samples = int(self.config['deadzone_ms'] * self.config['fs'] / 1000)

        # 6. Hysteresis search
        active = np.zeros(env_use.shape[1], dtype=bool)
        onsets = [[] for _ in range(env_use.shape[1])]

        for t in range(deadzone_samples, env_use.shape[0]):
            hi_cross = ~active & (env_use[t] > th_hi[t])
            lo_cross = active & (env_use[t] < th_lo[t])
            
            # Only append onsets that occur after the deadzone period
            for ch in np.where(hi_cross)[0]:
                onsets[ch].append(t)
            
            active = (active | hi_cross) & ~lo_cross

        if return_env:
            return onsets, env_use.squeeze()
        return onsets

    def _first_onsets_per_trial_offline(self, emg: np.ndarray, times: np.ndarray, trial_start_times: np.ndarray, trial_end_times: np.ndarray):
        """
        Detect first EMG onset per trial.
        
        Args:
            emg (np.ndarray): EMG signal data
            times (np.ndarray): Time array corresponding to EMG samples
            trial_start_times (np.ndarray): Array of times when each trial starts
            trial_end_times (np.ndarray): Array of times when each trial ends
        
        Returns:
            dict: Dictionary keyed by trial index with onset sample indices.
        """
        if emg.ndim != 2:
            raise ValueError("EMG data must be 2D (samples x channels)")
        
        if len(times) != emg.shape[0]:
            raise ValueError("times array must have same length as EMG data")
        
        trial_start_indices = np.searchsorted(times, trial_start_times)
        trial_end_indices = np.searchsorted(times, trial_end_times)
        
        onset_samples = []
        
        for trial_idx, (start, end) in enumerate(zip(trial_start_indices, trial_end_indices)):
            if start >= emg.shape[0] or end > emg.shape[0]:
                raise ValueError(f"Trial start/end indices out of bounds for trial {trial_idx}")
            
            # Extract trial data
            emg_trial = emg[start:end]
            
            if emg_trial.shape[0] == 0:
                onset_samples[trial_idx] = None
                continue
            
            # Detect onsets in this trial (deadzone is applied within _energy_cfar_onsets_offline)
            (onsets_list, env_trial) = self._energy_cfar_onsets_offline(
                emg_trial, return_env=True
            )

            if onsets_list and onsets_list[0]:  # at least one detection
                coarse = onsets_list[0][0]  # first onset, first (or only) channel
                if self.config['do_refine_onset']:
                    fine = self._refine_onset(env_trial, coarse)
                else:
                    fine = coarse
                # Convert to global index
                onset_samples.append(start + fine)
            else:  # no onset found
                onset_samples.append(None)

        return onset_samples

    def detect_movement_onsets(self, emg: np.ndarray, times: np.ndarray, trial_start_times: np.ndarray, trial_end_times: np.ndarray) -> np.ndarray:
        """
        Detect the onset of a movement based directly on the EMG.
        
        Args:
            emg (np.ndarray): EMG signal data, shape (samples, channels)
            times (np.ndarray): Time array corresponding to EMG samples
            trial_start_times (np.ndarray): Array of times when each trial starts
            trial_end_times (np.ndarray): Array of times when each trial ends
            
        Returns:
            np.ndarray: Array of onset times, shape (n_trials,). NaN where no onset detected.
                       Onsets are only detected after the deadzone period in each trial.
        """
        if len(times) != emg.shape[0]:
            raise ValueError("times array must have same length as EMG data")
        
        # Detect onsets using the configured parameters
        onsets_samples = self._first_onsets_per_trial_offline(
            emg=emg,
            times=times,
            trial_start_times=trial_start_times,
            trial_end_times=trial_end_times
        )
        
        n_trials = len(trial_start_times)
        
        # Convert onset sample indices to times
        onset_times = np.full(n_trials, np.nan)
        
        for trial_idx in range(n_trials):
            if onsets_samples[trial_idx] is not None:
                onset_sample_idx = onsets_samples[trial_idx]
                if 0 <= onset_sample_idx < len(times):
                    onset_times[trial_idx] = times[onset_sample_idx]
            
        return onset_times



