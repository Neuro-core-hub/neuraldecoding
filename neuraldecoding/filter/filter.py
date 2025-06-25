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

    def filter(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Apply causal filtering to input data.
        
        Parameters:
            data (np.ndarray): Input data to be filtered
            axis (int): The axis along which to apply the filter (default 0)
            
        Returns:
            np.ndarray: Filtered data
        """
        if self.use_sos:
            return signal.sosfilt(self.sos, data, axis=axis)
        else:
            return signal.lfilter(self.b, self.a, data, axis=axis)
    
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
    
    def apply(self, data: np.ndarray, causal: bool = True, axis: int = 0) -> np.ndarray:
        """
        Apply filtering to the data.
        
        Parameters:
            data (np.ndarray): Input data to be filtered
            causal (bool): Whether to use causal filtering (True) or non-causal filtering (False)
            axis (int): The axis along which to apply the filter (default 0)
            
        Returns:
            np.ndarray: Filtered data
        """
        if causal:
            return self.filter(data, axis=axis)
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
    print("Hello, world!")
    # Generate sample data: a signal with multiple frequency components
    fs = 1000  # Sampling frequency (Hz)
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second of data
    
    # Create a signal with components at 5 Hz, 50 Hz, and 200 Hz
    signal_clean = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 200 * t)
    
    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(t))
    signal_noisy = signal_clean + noise
    
    # Create a lowpass Butterworth filter to remove high frequencies
    lowpass_filter = GenericFilter(
        filter_type="butterworth",
        cutoff_freq=[30],  # 30 Hz cutoff
        order=4,
        fs=fs,
        btype="lowpass"
    )
    
    # Apply causal filtering (introduces phase lag)
    filtered_causal = lowpass_filter.filter(signal_noisy)
    
    # Apply non-causal filtering (zero phase distortion)
    filtered_noncausal = lowpass_filter.filtfilt(signal_noisy)
    
    # Create a bandpass filter to isolate the 50 Hz component
    bandpass_filter = GenericFilter(
        filter_type="butterworth",
        cutoff_freq=[40, 60],  # 40-60 Hz bandpass
        order=4,
        fs=fs,
        btype="bandpass"
    )
    
    # Apply the bandpass filter using the apply method
    filtered_bandpass = bandpass_filter.apply(signal_noisy, causal=False)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, signal_clean, label='Original Signal')
    plt.title('Original Signal')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(t, signal_noisy, label='Noisy Signal')
    plt.title('Noisy Signal')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, filtered_causal, label='Causal Filter')
    plt.plot(t, filtered_noncausal, label='Non-Causal Filter')
    plt.title('Filtered Signals')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
