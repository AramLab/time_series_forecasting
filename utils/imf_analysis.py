"""
IMF Statistical Analysis and Characteristics Module

Provides tools for analyzing Intrinsic Mode Functions (IMFs) from CEEMDAN decomposition,
computing their statistical properties, and determining optimal forecasting model based
on IMF characteristics.
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy as scipy_entropy
import warnings

warnings.filterwarnings('ignore')


class IMFAnalyzer:
    """Analyzes statistical characteristics of IMF components."""
    
    def __init__(self, imf):
        """
        Initialize IMF analyzer.
        
        Parameters:
        -----------
        imf : array-like
            Individual IMF component from CEEMDAN decomposition
        """
        self.imf = np.asarray(imf).flatten()
        self._compute_characteristics()
    
    def _compute_characteristics(self):
        """Compute all statistical characteristics of the IMF."""
        self.energy = np.sum(self.imf ** 2)
        self.mean = np.mean(self.imf)
        self.std = np.std(self.imf)
        self.var = np.var(self.imf)
        self.skewness = self._compute_skewness()
        self.kurtosis = self._compute_kurtosis()
        self.entropy = self._compute_entropy()
        self.frequency_features = self._compute_frequency_features()
    
    def _compute_skewness(self):
        """Compute skewness of IMF."""
        if self.std == 0:
            return 0
        third_moment = np.mean((self.imf - self.mean) ** 3)
        return third_moment / (self.std ** 3)
    
    def _compute_kurtosis(self):
        """Compute kurtosis of IMF."""
        if self.std == 0:
            return 0
        fourth_moment = np.mean((self.imf - self.mean) ** 4)
        return (fourth_moment / (self.std ** 4)) - 3  # Excess kurtosis
    
    def _compute_entropy(self):
        """
        Compute Shannon entropy of IMF to measure information content.
        Higher entropy = more noise/randomness; lower = more structure.
        """
        # Normalize IMF to [0, 1] range for histogram computation
        normalized = (self.imf - np.min(self.imf)) / (np.max(self.imf) - np.min(self.imf) + 1e-10)
        # Create histogram bins
        hist, _ = np.histogram(normalized, bins=50, range=(0, 1))
        # Normalize histogram to probability distribution
        hist = hist / np.sum(hist)
        # Remove zero bins for entropy calculation
        hist = hist[hist > 0]
        # Compute Shannon entropy
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    def _compute_frequency_features(self):
        """
        Compute frequency domain characteristics.
        
        Returns:
        --------
        dict with:
            - dominant_freq: frequency of maximum spectral power
            - freq_concentration: how concentrated power is at dominant frequency
            - spectral_entropy: entropy of power spectrum (normalized)
        """
        # Compute FFT
        fft = np.fft.fft(self.imf)
        power_spectrum = np.abs(fft) ** 2
        frequencies = np.fft.fftfreq(len(self.imf))
        
        # Get positive frequencies only
        pos_mask = frequencies > 0
        frequencies_pos = frequencies[pos_mask]
        power_pos = power_spectrum[pos_mask]
        
        # Dominant frequency
        if len(power_pos) > 0:
            dominant_freq_idx = np.argmax(power_pos)
            dominant_freq = frequencies_pos[dominant_freq_idx]
            
            # Frequency concentration: ratio of power at dominant freq to total power
            freq_concentration = power_pos[dominant_freq_idx] / (np.sum(power_pos) + 1e-10)
            
            # Spectral entropy (normalized)
            power_norm = power_pos / (np.sum(power_pos) + 1e-10)
            power_norm = power_norm[power_norm > 0]
            spec_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(power_norm))
            spec_entropy_norm = spec_entropy / max_entropy if max_entropy > 0 else 0
        else:
            dominant_freq = 0
            freq_concentration = 0
            spec_entropy_norm = 0
        
        return {
            'dominant_freq': dominant_freq,
            'freq_concentration': freq_concentration,
            'spectral_entropy': spec_entropy_norm
        }
    
    def get_characteristics(self):
        """
        Get all IMF characteristics.
        
        Returns:
        --------
        dict : All computed characteristics
        """
        return {
            'energy': self.energy,
            'mean': self.mean,
            'std': self.std,
            'variance': self.var,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'entropy': self.entropy,
            'dominant_freq': self.frequency_features['dominant_freq'],
            'freq_concentration': self.frequency_features['freq_concentration'],
            'spectral_entropy': self.frequency_features['spectral_entropy']
        }
    
    def is_noise_component(self, entropy_threshold=7.0, freq_concentration_threshold=0.15):
        """
        Determine if IMF is primarily noise component.
        
        Noise IMFs typically have:
        - High entropy (random)
        - Low frequency concentration (spread across spectrum)
        - Low autocorrelation
        
        Parameters:
        -----------
        entropy_threshold : float
            Threshold for entropy to classify as noise (0-max_entropy)
        freq_concentration_threshold : float
            Threshold for frequency concentration (0-1)
            
        Returns:
        --------
        bool : True if IMF appears to be noise
        """
        return (self.entropy > entropy_threshold and 
                self.frequency_features['freq_concentration'] < freq_concentration_threshold)
    
    def is_trend_component(self, std_threshold=0.05):
        """
        Determine if IMF is primarily trend component.
        
        Trend IMFs typically have:
        - High energy
        - Low frequency (dominant frequency near 0)
        - Smooth (low kurtosis)
        - Low entropy
        
        Parameters:
        -----------
        std_threshold : float
            Threshold for standard deviation normalization
            
        Returns:
        --------
        bool : True if IMF appears to be trend
        """
        # Trend should have low frequency content (near zero)
        is_low_freq = self.frequency_features['dominant_freq'] < 0.05
        # Trend should be relatively smooth (not noisy)
        is_smooth = self.entropy < 5.0
        # Trend should have relative stability (moderate skewness)
        is_stable = abs(self.skewness) < 2.0
        
        return is_low_freq and is_smooth and is_stable
    
    def is_seasonal_component(self, freq_concentration_threshold=0.3):
        """
        Determine if IMF is primarily seasonal component.
        
        Seasonal IMFs typically have:
        - Medium to high frequency concentration (periodic)
        - Regular oscillations
        - Moderate entropy
        
        Parameters:
        -----------
        freq_concentration_threshold : float
            Threshold for frequency concentration (0-1)
            
        Returns:
        --------
        bool : True if IMF appears to be seasonal
        """
        # Strong frequency concentration indicates periodicity
        is_periodic = self.frequency_features['freq_concentration'] > freq_concentration_threshold
        # Not too high entropy (not noise)
        is_not_noise = self.entropy < 7.0
        # Medium frequency (not trend, not very high-freq noise)
        is_mid_freq = 0.05 <= self.frequency_features['dominant_freq'] < 0.3
        
        return is_periodic and is_not_noise and is_mid_freq
    
    def classify(self):
        """
        Classify IMF into category: 'trend', 'seasonal', 'noise', or 'mixed'.
        
        Returns:
        --------
        str : Classification of IMF
        """
        if self.is_trend_component():
            return 'trend'
        elif self.is_seasonal_component():
            return 'seasonal'
        elif self.is_noise_component():
            return 'noise'
        else:
            return 'mixed'


def analyze_imf_components(imfs, verbose=False):
    """
    Analyze all IMF components from CEEMDAN decomposition.
    
    Parameters:
    -----------
    imfs : list or array-like
        List of IMF components
    verbose : bool
        If True, print analysis summary
        
    Returns:
    --------
    list : List of IMFAnalyzer objects
    """
    analyzers = []
    for i, imf in enumerate(imfs):
        analyzer = IMFAnalyzer(imf)
        analyzers.append(analyzer)
        
        if verbose:
            chars = analyzer.get_characteristics()
            classification = analyzer.classify()
            print(f"\nIMF {i}: {classification}")
            print(f"  Energy: {chars['energy']:.4e}")
            print(f"  Entropy: {chars['entropy']:.4f}")
            print(f"  Spectral Entropy: {chars['spectral_entropy']:.4f}")
            print(f"  Freq Concentration: {chars['freq_concentration']:.4f}")
    
    return analyzers


def compute_imf_weights(imfs, method='energy'):
    """
    Compute adaptive weights for IMF components.
    
    Parameters:
    -----------
    imfs : list or array-like
        List of IMF components
    method : str
        Method for computing weights:
        - 'energy': Weight by IMF energy (normalized)
        - 'entropy': Weight inversely by entropy (lower entropy = higher weight)
        - 'forecast_error': Would need forecast errors (not implemented here)
        
    Returns:
    --------
    array : Normalized weights for each IMF
    """
    if method == 'energy':
        energies = np.array([np.sum(imf ** 2) for imf in imfs])
        weights = energies / (np.sum(energies) + 1e-10)
    elif method == 'entropy':
        analyzers = analyze_imf_components(imfs)
        entropies = np.array([a.entropy for a in analyzers])
        # Invert entropy: lower entropy gets higher weight
        weights = 1.0 / (entropies + 1e-10)
        weights = weights / np.sum(weights)
    else:
        # Default: uniform weights
        weights = np.ones(len(imfs)) / len(imfs)
    
    return weights


def get_model_recommendation_for_imf(imf, detailed=False):
    """
    Get recommended forecasting model(s) for a specific IMF based on its characteristics.
    
    Parameters:
    -----------
    imf : array-like
        Individual IMF component
    detailed : bool
        If True, return scores for all models; if False, return just best model
        
    Returns:
    --------
    str or dict : Recommended model name(s)
        If detailed=False: str with model name ('ARIMA', 'ETS', 'LSTM', 'Prophet')
        If detailed=True: dict with scores for each model
    """
    analyzer = IMFAnalyzer(imf)
    classification = analyzer.classify()
    chars = analyzer.get_characteristics()
    
    # Base scores for each model
    scores = {
        'ARIMA': 0.0,
        'ETS': 0.0,
        'LSTM': 0.0,
        'Prophet': 0.0
    }
    
    # Classification-based scoring
    if classification == 'trend':
        scores['ARIMA'] += 3.0
        scores['ETS'] += 2.5
        scores['Prophet'] += 2.0
        scores['LSTM'] += 1.0
    elif classification == 'seasonal':
        scores['ARIMA'] += 3.0
        scores['ETS'] += 3.0
        scores['Prophet'] += 2.5
        scores['LSTM'] += 2.0
    elif classification == 'noise':
        scores['LSTM'] += 2.5
        scores['ARIMA'] += 1.5
        scores['ETS'] += 1.0
        scores['Prophet'] += 0.5
    else:  # mixed
        scores['LSTM'] += 2.0
        scores['ARIMA'] += 2.0
        scores['ETS'] += 1.5
        scores['Prophet'] += 1.5
    
    # Entropy-based adjustment
    if chars['entropy'] < 4.0:
        scores['ARIMA'] += 1.0
        scores['Prophet'] += 0.5
    elif chars['entropy'] > 7.0:
        scores['LSTM'] += 1.5
    
    # Frequency-based adjustment
    freq_conc = chars['freq_concentration']
    if freq_conc > 0.5:
        scores['ETS'] += 1.0
        scores['Prophet'] += 1.0
    elif freq_conc < 0.2:
        scores['LSTM'] += 0.5
    
    # Normalize scores
    total_score = sum(scores.values())
    if total_score > 0:
        scores = {k: v / total_score for k, v in scores.items()}
    
    if detailed:
        return scores
    else:
        # Return model with highest score
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        return best_model
