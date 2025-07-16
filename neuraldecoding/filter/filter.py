import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class GenericFilter:
    def __init__(self, filter_type: str, cutoff_freq: list, order: int, fs: float = 1.0, 
                 btype: str = 'lowpass', use_sos: bool = True, **filter_kwargs):
        """
        Initialize a generic filter.
        
        Parameters:
            filter_type (str): Type of filter ('butterworth', 'chebyshev1', 'chebyshev2', 'bessel', 'elliptic')
            cutoff_freq (list): List of cutoff frequencies in Hz
                - For lowpass/highpass: [freq]
                - For bandpass/bandstop: [low_freq, high_freq]
            order (int): Filter order
            fs (float): Sampling frequency in Hz (default 1.0)
            btype (str): Band type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'
            use_sos (bool): Use second-order sections for filter implementation (more stable for high-order filters)
            **filter_kwargs: Additional filter-specific parameters:
                - rp: maximum ripple in the passband (dB) (for Chebyshev I and elliptic)
                - rs: minimum attenuation in the stopband (dB) (for Chebyshev II and elliptic)
                - analog: design an analog filter (default: False)
        """
        self.filter_type = filter_type.lower()
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.fs = fs
        self.btype = btype
        self.use_sos = use_sos
        self.filter_kwargs = filter_kwargs
        
        self.normalized_freq = [f / (self.fs / 2) for f in self.cutoff_freq]
        
        # Default filter coefficients
        self.b, self.a = None, None
        self.sos = None
        
        # Design the filter
        self._design_filter()

    def _design_filter(self):
        """
        Design the filter based on specified parameters.
        """
        if self.btype in ['lowpass', 'highpass'] and len(self.cutoff_freq) != 1:
            raise ValueError(f"{self.btype} filter requires exactly one cutoff frequency")
        
        if self.btype in ['bandpass', 'bandstop'] and len(self.cutoff_freq) != 2:
            raise ValueError(f"{self.btype} filter requires exactly two cutoff frequencies")
        
        # Get normalized frequency/frequencies
        Wn = self.normalized_freq[0] if len(self.normalized_freq) == 1 else self.normalized_freq
        
        # Extract common filter kwargs
        analog = self.filter_kwargs.get('analog', False)
        output = 'sos' if self.use_sos else 'ba'
        
        # Design filter based on type
        if self.filter_type == "butterworth":
            if self.use_sos:
                self.sos = signal.butter(self.order, Wn, btype=self.btype, analog=analog, output='sos')
            else:
                self.b, self.a = signal.butter(self.order, Wn, btype=self.btype, analog=analog, output='ba')
                
        elif self.filter_type == "chebyshev1":
            rp = self.filter_kwargs.get('rp', 0.5)  # Default passband ripple
            if self.use_sos:
                self.sos = signal.cheby1(self.order, rp, Wn, btype=self.btype, analog=analog, output='sos')
            else:
                self.b, self.a = signal.cheby1(self.order, rp, Wn, btype=self.btype, analog=analog, output='ba')
                
        elif self.filter_type == "chebyshev2":
            rs = self.filter_kwargs.get('rs', 40)  # Default stopband attenuation
            if self.use_sos:
                self.sos = signal.cheby2(self.order, rs, Wn, btype=self.btype, analog=analog, output='sos')
            else:
                self.b, self.a = signal.cheby2(self.order, rs, Wn, btype=self.btype, analog=analog, output='ba')
                
        elif self.filter_type == "bessel":
            norm = self.filter_kwargs.get('norm', 'phase')  # Default normalization
            if self.use_sos:
                self.sos = signal.bessel(self.order, Wn, btype=self.btype, analog=analog, output='sos', norm=norm)
            else:
                self.b, self.a = signal.bessel(self.order, Wn, btype=self.btype, analog=analog, output='ba', norm=norm)
                
        elif self.filter_type == "elliptic":
            rp = self.filter_kwargs.get('rp', 0.5)  # Default passband ripple
            rs = self.filter_kwargs.get('rs', 40)   # Default stopband attenuation
            if self.use_sos:
                self.sos = signal.ellip(self.order, rp, rs, Wn, btype=self.btype, analog=analog, output='sos')
            else:
                self.b, self.a = signal.ellip(self.order, rp, rs, Wn, btype=self.btype, analog=analog, output='ba')
                
        else:
            raise ValueError(f"Filter type {self.filter_type} not supported")

    def filter(self, data: np.ndarray, axis: int = 0, zi=None) -> tuple:
        """
        Apply causal filtering to input data with state handling for real-time processing.
        
        Parameters:
            data (np.ndarray): Input data to be filtered
            axis (int): The axis along which to apply the filter (default 0)
            zi (ndarray, optional): Initial filter states. If None, zeros are used.
                For SOS filters: shape should be (n_sections, 2, n_channels) for axis=0
                For BA filters: shape should be (n_order, n_channels) for axis=0
            
        Returns:
            tuple: (filtered_data, z_final)
                - filtered_data (np.ndarray): Filtered data with same shape as input
                - z_final (ndarray): Final filter states for use in subsequent calls
        """
        if self.use_sos:
            if zi is None:
                # Filter without initial state
                filtered_data = signal.sosfilt(self.sos, data, axis=axis)
                # Get the state that would result from filtering this data
                n_sections = self.sos.shape[0]
                n_channels = data.shape[1] if data.ndim > 1 else 1
                zi = signal.sosfilt_zi(self.sos)[:, :, np.newaxis]  # (n_sections, 2, 1)
                
                # Repeat for each channel if multichannel data
                if n_channels > 1:
                    zi = np.repeat(zi, n_channels, axis=2)  # (n_sections, 2, n_channels)
                
                # Run a small portion of the data to get the final state
                if data.size > 0:
                    _, z_final = signal.sosfilt(self.sos, data, axis=axis, zi=zi)
                else:
                    z_final = zi
                    
                return filtered_data, z_final
            else:
                # Filter with provided initial state
                return signal.sosfilt(self.sos, data, axis=axis, zi=zi)
        else:
            if zi is None:
                # Filter without initial state
                filtered_data = signal.lfilter(self.b, self.a, data, axis=axis)
                
                # Create and return final state
                n_order = max(len(self.a) - 1, 0)
                n_channels = data.shape[1] if data.ndim > 1 else 1
                
                # Get the zi shape right
                zi = signal.lfilter_zi(self.b, self.a)  # (n_order,)
                
                # Reshape for multi-channel if needed
                if n_channels > 1:
                    zi = np.repeat(zi[:, np.newaxis], n_channels, axis=1)  # (n_order, n_channels)
                
                # Run a small portion of the data to get the final state
                if data.size > 0:
                    _, z_final = signal.lfilter(self.b, self.a, data, axis=axis, zi=zi)
                else:
                    z_final = zi
                    
                return filtered_data, z_final
            else:
                # Filter with provided initial state
                return signal.lfilter(self.b, self.a, data, axis=axis, zi=zi)
    
    def filtfilt(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Apply non-causal, zero-phase filtering to input data.
        
        Parameters:
            data (np.ndarray): Input data to be filtered
            axis (int): The axis along which to apply the filter (default 0)
            
        Returns:
            np.ndarray: Filtered data with zero phase distortion
        """
        if self.use_sos:
            return signal.sosfiltfilt(self.sos, data, axis=axis)
        else:
            return signal.filtfilt(self.b, self.a, data, axis=axis)
    
    def apply(self, data: np.ndarray, causal: bool = True, axis: int = 0, zi=None) -> tuple:
        """
        Apply filtering to the data.
        
        Parameters:
            data (np.ndarray): Input data to be filtered
            causal (bool): Whether to use causal filtering (True) or non-causal filtering (False)
            axis (int): The axis along which to apply the filter (default 0)
            zi (ndarray, optional): Initial filter states for causal filtering. 
                                    Ignored for non-causal filtering.
            
        Returns:
            tuple or np.ndarray: 
                - For causal filtering: (filtered_data, z_final)
                - For non-causal filtering: filtered_data only
        """
        if causal:
            return self.filter(data, axis=axis, zi=zi)
        else:
            return self.filtfilt(data, axis=axis)
    
    def get_frequency_response(self, n_points: int = 1000) -> tuple:
        """
        Calculate the frequency response of the filter.
        
        Parameters:
            n_points (int): Number of frequency points to calculate
            
        Returns:
            tuple: (frequencies, response magnitude in dB)
        """
        w, h = signal.freqz(self.b, self.a, worN=n_points) if not self.use_sos else signal.sosfreqz(self.sos, worN=n_points)
        freq = w * self.fs / (2 * np.pi)
        magnitude_db = 20 * np.log10(abs(h))
        return freq, magnitude_db


# Example usage
if __name__ == "__main__":
    # Generate sample data: a simple signal with multiple frequency components
    fs = 1000  # Sampling frequency (Hz)
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second of data
    
    # Create a signal with 5 Hz and 50 Hz components
    clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    
    # Add some noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, size=clean_signal.shape)
    signal_noisy = clean_signal + noise
    
    # Create a lowpass Butterworth filter to remove high frequencies
    lowpass_filter = GenericFilter(
        filter_type="butterworth",
        cutoff_freq=[20],  # 20 Hz cutoff
        order=4,
        fs=fs,
        btype="lowpass"
    )
    
    # Apply the filter
    # Reshape to 2D for consistent handling (add channel dimension)
    signal_noisy_2d = signal_noisy.reshape(-1, 1)
    
    # Apply filters
    filtered_causal_2d, _ = lowpass_filter.filter(signal_noisy_2d)  # Causal filtering
    filtered_noncausal_2d = lowpass_filter.filtfilt(signal_noisy_2d)  # Non-causal filtering
    
    # Convert back to 1D
    filtered_causal = filtered_causal_2d.flatten()
    filtered_noncausal = filtered_noncausal_2d.flatten()
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, clean_signal)
    plt.title('Original Clean Signal')
    
    plt.subplot(3, 1, 2)
    plt.plot(t, signal_noisy)
    plt.title('Noisy Signal')
    
    plt.subplot(3, 1, 3)
    plt.plot(t, filtered_causal, label='Causal Filter')
    plt.plot(t, filtered_noncausal, label='Non-causal Filter')
    plt.title('Filtered Signals')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot frequency response of the filter
    freq, magnitude_db = lowpass_filter.get_frequency_response()
    
    plt.figure(figsize=(8, 4))
    plt.plot(freq, magnitude_db)
    plt.axvline(20, color='r', linestyle='--', label='Cutoff (20 Hz)')
    plt.title('Filter Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
