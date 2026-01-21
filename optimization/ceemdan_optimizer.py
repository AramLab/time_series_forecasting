"""
CEEMDAN Parameter Optimization Module

Implements optimization algorithms for finding optimal CEEMDAN hyperparameters
(trials and noise_width) based on decomposition quality criteria.
"""

import numpy as np
from itertools import product
import warnings
from scipy.stats import variation

warnings.filterwarnings('ignore')


class CEEMDANOptimizer:
    """Optimizer for CEEMDAN parameters."""
    
    def __init__(self, decompose_func, series, test_size=0.2):
        """
        Initialize CEEMDAN optimizer.
        
        Parameters:
        -----------
        decompose_func : callable
            Function that performs CEEMDAN decomposition.
            Signature: decompose_func(series, trials, noise_width) -> imfs
        series : array-like
            Time series to optimize decomposition for
        test_size : float
            Fraction of series to use for test (if applicable)
        """
        self.decompose_func = decompose_func
        self.series = np.asarray(series)
        self.test_size = test_size
    
    def compute_decomposition_quality(self, imfs):
        """
        Compute quality metrics for CEEMDAN decomposition.
        
        Parameters:
        -----------
        imfs : list or array
            Intrinsic Mode Functions from CEEMDAN
            
        Returns:
        --------
        dict : Quality metrics
            - orthogonality: Measure of IMF orthogonality (higher is better)
            - energy_separation: How well energy is distributed (0-1, higher is better)
            - residual_trend: Whether residual is monotonic trend
            - imf_count: Number of IMFs
        """
        imfs = [np.asarray(imf) for imf in imfs]
        n_imfs = len(imfs)
        
        # 1. Orthogonality Index: measures independence between IMFs
        # Lower values indicate better orthogonality
        orthogonality = self._compute_orthogonality_index(imfs)
        
        # 2. Energy Separation: measures how well energy is distributed
        # across IMFs (avoid single dominant IMF)
        energy_separation = self._compute_energy_separation(imfs)
        
        # 3. Residual Trend: residual should be monotonic (0-1 scale)
        residual = imfs[-1] if n_imfs > 0 else np.zeros_like(self.series)
        residual_trend_score = self._is_monotonic_trend(residual)
        
        return {
            'orthogonality': orthogonality,
            'energy_separation': energy_separation,
            'residual_trend': residual_trend_score,
            'imf_count': n_imfs
        }
    
    def _compute_orthogonality_index(self, imfs):
        """
        Compute Orthogonality Index (OI) for IMFs.
        
        OI measures cross-correlation between different IMFs.
        Lower OI indicates better orthogonality.
        
        Formula: OI = sum(|<IMFi, IMFj>| / (||IMFi|| * ||IMFj||)) for i != j
        """
        if len(imfs) < 2:
            return 0.0
        
        oi = 0.0
        n_imfs = len(imfs)
        
        for i in range(n_imfs):
            for j in range(i + 1, n_imfs):
                # Cross-correlation
                cross_corr = np.abs(np.dot(imfs[i], imfs[j]))
                # Normalization
                norm_i = np.sqrt(np.dot(imfs[i], imfs[i]))
                norm_j = np.sqrt(np.dot(imfs[j], imfs[j]))
                
                if norm_i > 0 and norm_j > 0:
                    oi += cross_corr / (norm_i * norm_j)
        
        # Normalize by number of pairs
        n_pairs = n_imfs * (n_imfs - 1) / 2
        oi = oi / max(n_pairs, 1)
        
        return oi
    
    def _compute_energy_separation(self, imfs):
        """
        Compute energy separation metric.
        
        Good decomposition has energy distributed across IMFs.
        Returns normalized entropy of energy distribution (0-1).
        Lower is better (concentrated energy) but should have some spread.
        Optimal range: 0.4-0.8
        """
        energies = np.array([np.sum(imf ** 2) for imf in imfs])
        total_energy = np.sum(energies)
        
        if total_energy == 0:
            return 0.0
        
        # Normalize energies to probability distribution
        energy_dist = energies / total_energy
        energy_dist = energy_dist[energy_dist > 0]
        
        # Entropy of energy distribution
        entropy = -np.sum(energy_dist * np.log2(energy_dist + 1e-10))
        # Normalize by maximum entropy
        max_entropy = np.log2(len(energy_dist))
        if max_entropy > 0:
            entropy_norm = entropy / max_entropy
        else:
            entropy_norm = 0.0
        
        # Score: we want some spread but not too much
        # Optimal entropy ~0.7 of maximum
        score = 1.0 - np.abs(entropy_norm - 0.7)
        return max(0.0, score)
    
    def _is_monotonic_trend(self, residual):
        """
        Check if residual follows monotonic trend (0-1 score).
        
        Residual should be monotonic or have few turning points.
        Score: 1 - (turning_points / max_possible_turning_points)
        """
        residual = np.asarray(residual)
        diffs = np.diff(residual)
        
        if len(diffs) == 0:
            return 1.0
        
        # Count sign changes (turning points)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        max_turning_points = len(diffs) - 1
        
        if max_turning_points == 0:
            score = 1.0
        else:
            score = 1.0 - (sign_changes / max_turning_points)
        
        return max(0.0, score)
    
    def grid_search(self, trials_range=(10, 100, 10), 
                    noise_width_range=(0.01, 0.20, 0.02)):
        """
        Perform grid search for optimal CEEMDAN parameters.
        
        Parameters:
        -----------
        trials_range : tuple
            (min_trials, max_trials, step)
        noise_width_range : tuple
            (min_noise_width, max_noise_width, step)
            
        Returns:
        --------
        dict : Results with best parameters and scores
        """
        trials_list = np.arange(trials_range[0], trials_range[1] + 1, trials_range[2]).astype(int)
        noise_widths = np.arange(noise_width_range[0], noise_width_range[1] + noise_width_range[2] / 2, noise_width_range[2])
        
        results = []
        best_quality = -np.inf
        best_params = None
        
        for trials, noise_width in product(trials_list, noise_widths):
            try:
                imfs = self.decompose_func(self.series, trials=trials, noise_width=noise_width)
                quality = self.compute_decomposition_quality(imfs)
                
                # Combined quality score
                combined_score = (
                    -quality['orthogonality'] +  # Minimize orthogonality index
                    quality['energy_separation'] +  # Maximize energy separation
                    quality['residual_trend']  # Maximize residual trend quality
                )
                
                results.append({
                    'trials': trials,
                    'noise_width': noise_width,
                    'quality': quality,
                    'combined_score': combined_score
                })
                
                if combined_score > best_quality:
                    best_quality = combined_score
                    best_params = {'trials': trials, 'noise_width': noise_width}
            
            except Exception as e:
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_quality,
            'all_results': results,
            'search_space': {
                'trials': trials_list,
                'noise_width': noise_widths
            }
        }
    
    def bayesian_search(self, n_iterations=20, initial_points=5):
        """
        Perform Bayesian optimization for CEEMDAN parameters.
        
        Note: This is a simplified version without external library.
        For production use, consider using skopt or optuna.
        
        Parameters:
        -----------
        n_iterations : int
            Number of optimization iterations
        initial_points : int
            Number of random initial evaluations
            
        Returns:
        --------
        dict : Results with best parameters and history
        """
        from scipy.optimize import minimize
        
        # Search bounds
        bounds = [(10, 100), (0.01, 0.20)]
        
        # Objective function to minimize
        def objective(params):
            trials = int(params[0])
            noise_width = params[1]
            
            try:
                imfs = self.decompose_func(self.series, trials=trials, noise_width=noise_width)
                quality = self.compute_decomposition_quality(imfs)
                
                # Negative combined score (we want to maximize)
                combined_score = (
                    -quality['orthogonality'] +
                    quality['energy_separation'] +
                    quality['residual_trend']
                )
                
                return -combined_score  # Negative because optimizer minimizes
            except:
                return 1e10  # Large penalty for failed decompositions
        
        # Random search for initial points
        best_score = np.inf
        best_params = None
        history = []
        
        for _ in range(initial_points):
            trials = np.random.randint(10, 101)
            noise_width = np.random.uniform(0.01, 0.20)
            score = objective([trials, noise_width])
            
            history.append({
                'trials': trials,
                'noise_width': noise_width,
                'score': -score
            })
            
            if score < best_score:
                best_score = score
                best_params = [trials, noise_width]
        
        # Local optimization from best random point
        if best_params is not None:
            result = minimize(
                objective,
                x0=best_params,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': n_iterations - initial_points}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        
        return {
            'best_params': {
                'trials': int(best_params[0]),
                'noise_width': float(best_params[1])
            } if best_params is not None else None,
            'best_score': -best_score if best_params is not None else np.nan,
            'optimization_history': history
        }


def optimize_ceemdan_params(decompose_func, series, method='grid', 
                           verbose=False, **kwargs):
    """
    Find optimal CEEMDAN parameters for a time series.
    
    Parameters:
    -----------
    decompose_func : callable
        CEEMDAN decomposition function
    series : array-like
        Time series to optimize for
    method : str
        Optimization method: 'grid' or 'bayesian'
    verbose : bool
        Print optimization progress
    **kwargs : dict
        Additional parameters for optimization method
        
    Returns:
    --------
    dict : Best parameters with optimization details
    """
    optimizer = CEEMDANOptimizer(decompose_func, series)
    
    if method == 'grid':
        trials_range = kwargs.get('trials_range', (10, 100, 10))
        noise_width_range = kwargs.get('noise_width_range', (0.01, 0.20, 0.02))
        
        if verbose:
            print(f"Starting grid search over:")
            print(f"  trials: {trials_range[0]}-{trials_range[1]} (step {trials_range[2]})")
            print(f"  noise_width: {noise_width_range[0]:.3f}-{noise_width_range[1]:.3f} (step {noise_width_range[2]:.3f})")
        
        results = optimizer.grid_search(trials_range, noise_width_range)
    
    elif method == 'bayesian':
        n_iterations = kwargs.get('n_iterations', 20)
        
        if verbose:
            print(f"Starting Bayesian optimization ({n_iterations} iterations)...")
        
        results = optimizer.bayesian_search(n_iterations)
    
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    if verbose and 'all_results' in results:
        print(f"\nOptimization complete.")
        print(f"Best parameters: {results['best_params']}")
        print(f"Best score: {results['best_score']:.4f}")
    
    return results
