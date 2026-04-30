"""
Shared Utilities for Baseline Probabilistic Methods

Contains:
- Data loading functions
- Evaluation metrics (BCOR, RMSE, CRPS, Coverage)
- Loss functions for training
- Quantile conversion utilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os
import random
from typing import List, Tuple, Dict, Optional

# ============================================================================
# Seed for Reproducibility
# ============================================================================

SEED = 42

def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Constants
# ============================================================================

# Auto-detect repository root from file location
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))  # Go up 2 levels to repo root
BASE_DIR = os.path.dirname(_CURRENT_DIR)  # baselines/ directory
UNIFIED_DATA_DIR = os.path.join(REPO_ROOT, 'data')

QUANTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
N_QUANTILES = len(QUANTILE_LEVELS)

# Dataset configurations (using unified data from paper_model_sqr_unified)
DATASET_CONFIGS = {
    'BoM': {
        'data_dir': os.path.join(UNIFIED_DATA_DIR, 'BOM'),
        'n_lead_days': 62,
        'n_members': 33,
        'test_years': list(range(1999, 2014)),  # 15 folds
        'all_years': list(range(1981, 2014)),
        'train_window': 15,
    },
    'JMA': {
        'data_dir': os.path.join(UNIFIED_DATA_DIR, 'JMA'),
        'n_lead_days': 33,
        'n_members': 5,
        'test_years': list(range(1998, 2013)),  # 15 folds
        'all_years': list(range(1981, 2013)),
        'train_window': 15,
    },
    'CNRM': {
        'data_dir': os.path.join(UNIFIED_DATA_DIR, 'CNRM'),
        'n_lead_days': 60,
        'n_members': 15,
        'test_years': list(range(2010, 2015)),  # 5 folds
        'all_years': list(range(1993, 2015)),
        'train_window': 15,
    },
}

# Legacy BOM config for backward compatibility
BOM_CONFIG = DATASET_CONFIGS['BoM']

# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(dataset_name: str = 'BoM') -> Dict[str, np.ndarray]:
    """
    Load data for a specific dataset from unified data directory.

    Args:
        dataset_name: 'BoM', 'JMA', or 'CNRM'

    Returns:
        Dictionary with keys:
        - 'forecast_rmm1': (n_samples, n_lead_days+1) - forecast with day0
        - 'forecast_rmm2': (n_samples, n_lead_days+1)
        - 'gt_rmm1': (n_samples, n_lead_days) - ground truth
        - 'gt_rmm2': (n_samples, n_lead_days)
        - 'amplitude_forecast': (n_samples, n_lead_days)
        - 'day0_info': (n_samples, 4) - [rmm1, rmm2, phase, amplitude]
        - 'times': (n_samples,) - datetime array
        - 'mean_rmm1': (n_samples, n_lead_days) - computed ensemble mean (for lead days 1+)
        - 'mean_rmm2': (n_samples, n_lead_days)
        - 'std_rmm1': (n_samples, n_lead_days) - placeholder (use amplitude_std)
        - 'std_rmm2': (n_samples, n_lead_days)
        - 'amplitude_mean': (n_samples, n_lead_days)
        - 'amplitude_std': (n_samples, n_lead_days)
    """
    config = DATASET_CONFIGS[dataset_name]
    data_dir = config['data_dir']
    n_lead_days = config['n_lead_days']

    # Load raw data
    forecast_rmm1 = np.load(os.path.join(data_dir, 'forecast_rmm1_with_day0.npy'))
    forecast_rmm2 = np.load(os.path.join(data_dir, 'forecast_rmm2_with_day0.npy'))
    gt_rmm1 = np.load(os.path.join(data_dir, 'ground_truth_rmm1.npy'))
    gt_rmm2 = np.load(os.path.join(data_dir, 'ground_truth_rmm2.npy'))
    amplitude_forecast = np.load(os.path.join(data_dir, 'amplitude_forecast.npy'))
    day0_info = np.load(os.path.join(data_dir, 'day0_info.npy'))
    times = np.load(os.path.join(data_dir, 'times.npy'), allow_pickle=True)

    # For these methods, we use ensemble mean = forecast values (since it's already the mean)
    # Extract lead days 1 to n_lead_days (exclude day0 which is index 0)
    mean_rmm1 = forecast_rmm1[:, 1:n_lead_days+1]
    mean_rmm2 = forecast_rmm2[:, 1:n_lead_days+1]

    # Amplitude from day0 and forecast
    day0_amplitude = day0_info[:, 3]
    amplitude_full = np.zeros((len(forecast_rmm1), n_lead_days + 1))
    amplitude_full[:, 0] = day0_amplitude
    amplitude_full[:, 1:] = amplitude_forecast[:, :n_lead_days]

    amplitude_mean = amplitude_full[:, 1:n_lead_days+1]

    # For spread, we use a proxy based on amplitude (larger amplitude = more uncertainty)
    # This is a simplification - in reality we'd need ensemble members
    amplitude_std = np.abs(amplitude_mean) * 0.1 + 0.1  # Heuristic spread

    # Use amplitude-based spread for RMM components too
    std_rmm1 = amplitude_std
    std_rmm2 = amplitude_std

    data = {
        'forecast_rmm1': forecast_rmm1,
        'forecast_rmm2': forecast_rmm2,
        'gt_rmm1': gt_rmm1,
        'gt_rmm2': gt_rmm2,
        'amplitude_forecast': amplitude_forecast,
        'day0_info': day0_info,
        'times': times,
        'mean_rmm1': mean_rmm1,
        'mean_rmm2': mean_rmm2,
        'std_rmm1': std_rmm1,
        'std_rmm2': std_rmm2,
        'amplitude_mean': amplitude_mean,
        'amplitude_std': amplitude_std,
    }
    return data


def load_bom_data() -> Dict[str, np.ndarray]:
    """Legacy function for backward compatibility."""
    return load_dataset('BoM')


def load_dataset_with_ensemble(dataset_name: str = 'BoM') -> Dict[str, np.ndarray]:
    """
    Load dataset with full ensemble members.

    Data source: paper_model_sqr_direct/data/{BOM,JMA,CNRM}/

    Returns:
        Dictionary with keys:
        - 'ensemble_rmm1': (n_samples, n_lead_days, n_members) - all ensemble members
        - 'ensemble_rmm2': (n_samples, n_lead_days, n_members)
        - 'mean_rmm1': (n_samples, n_lead_days) - ensemble mean
        - 'mean_rmm2': (n_samples, n_lead_days)
        - 'std_rmm1': (n_samples, n_lead_days) - ensemble std
        - 'std_rmm2': (n_samples, n_lead_days)
        - 'gt_rmm1': (n_samples, n_lead_days)
        - 'gt_rmm2': (n_samples, n_lead_days)
        - 'amplitude_std': (n_samples, n_lead_days)
        - 'times': (n_samples,)
    """
    config = DATASET_CONFIGS[dataset_name]
    n_lead_days = config['n_lead_days']

    # Map dataset name to directory name
    dir_name = 'BOM' if dataset_name == 'BoM' else dataset_name
    data_dir = os.path.join(UNIFIED_DATA_DIR, dir_name)

    # Load ensemble members (different naming for BoM vs JMA/CNRM)
    if dataset_name == 'BoM':
        ensemble_rmm1 = np.load(os.path.join(data_dir, 'rmm1_ensemble.npy'))
        ensemble_rmm2 = np.load(os.path.join(data_dir, 'rmm2_ensemble.npy'))
    else:
        ensemble_rmm1 = np.load(os.path.join(data_dir, 'ensemble_rmm1.npy'))
        ensemble_rmm2 = np.load(os.path.join(data_dir, 'ensemble_rmm2.npy'))

    # Load ground truth
    gt_rmm1 = np.load(os.path.join(data_dir, 'ground_truth_rmm1.npy'))
    gt_rmm2 = np.load(os.path.join(data_dir, 'ground_truth_rmm2.npy'))
    times = np.load(os.path.join(data_dir, 'times.npy'), allow_pickle=True)

    # Compute mean and std from ensemble
    mean_rmm1 = np.mean(ensemble_rmm1, axis=-1)
    mean_rmm2 = np.mean(ensemble_rmm2, axis=-1)
    std_rmm1 = np.std(ensemble_rmm1, axis=-1)
    std_rmm2 = np.std(ensemble_rmm2, axis=-1)

    # Compute amplitude from ensemble mean
    amplitude_mean = np.sqrt(mean_rmm1**2 + mean_rmm2**2)

    # Use ensemble std as amplitude_std proxy
    amplitude_std = np.sqrt(std_rmm1**2 + std_rmm2**2)

    data = {
        'ensemble_rmm1': ensemble_rmm1,
        'ensemble_rmm2': ensemble_rmm2,
        'mean_rmm1': mean_rmm1,
        'mean_rmm2': mean_rmm2,
        'std_rmm1': std_rmm1,
        'std_rmm2': std_rmm2,
        'gt_rmm1': gt_rmm1,
        'gt_rmm2': gt_rmm2,
        'amplitude_mean': amplitude_mean,
        'amplitude_std': amplitude_std,
        'times': times,
    }
    return data


def load_bom_ensemble_data() -> Dict[str, np.ndarray]:
    """Legacy function for backward compatibility."""
    return load_dataset_with_ensemble('BoM')


def get_year_indices(year: int, all_years: List[int] = None) -> Tuple[int, int]:
    """
    Get start and end indices for a specific year in the dataset.

    BOM data has daily forecasts, ~365 samples per year.
    Total samples: 2375 for years 1999-2013 (15 test years from larger dataset)
    """
    if all_years is None:
        all_years = BOM_CONFIG['all_years']

    # Approximate samples per year (varies due to leap years, missing data)
    # For simplicity, we compute based on actual year boundaries
    # The data is ordered chronologically

    # Load times to get actual year boundaries
    times_file = os.path.join(DATA_DIR, 'times.npy')
    if os.path.exists(times_file):
        times = np.load(times_file, allow_pickle=True)
        # times contains datetime objects or similar
        # Extract year from each time
        year_mask = np.array([t.year == year if hasattr(t, 'year') else False for t in times])
        if year_mask.any():
            indices = np.where(year_mask)[0]
            return indices[0], indices[-1] + 1

    # Fallback: approximate based on position
    # Assuming roughly equal samples per year
    test_years = BOM_CONFIG['test_years']
    if year not in test_years:
        raise ValueError(f"Year {year} not in test years {test_years}")

    year_idx = test_years.index(year)
    n_total = 2375
    n_years = len(test_years)
    samples_per_year = n_total // n_years

    start = year_idx * samples_per_year
    end = start + samples_per_year if year_idx < n_years - 1 else n_total

    return start, end


def split_data_by_years(data: Dict[str, np.ndarray],
                        train_years: List[int],
                        valid_years: List[int],
                        test_years: List[int]) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into train/valid/test sets by year.

    Returns:
        train_data, valid_data, test_data - each is a dict with same keys as input
    """
    def get_indices_for_years(years):
        indices = []
        for year in years:
            try:
                start, end = get_year_indices(year)
                indices.extend(range(start, end))
            except ValueError:
                continue
        return indices

    train_idx = get_indices_for_years(train_years)
    valid_idx = get_indices_for_years(valid_years)
    test_idx = get_indices_for_years(test_years)

    def subset_data(data, indices):
        return {k: v[indices] if len(indices) > 0 else v[:0] for k, v in data.items()}

    return (subset_data(data, train_idx),
            subset_data(data, valid_idx),
            subset_data(data, test_idx))


# ============================================================================
# Evaluation Metrics (NumPy - for final evaluation)
# ============================================================================

def bcor(p1: np.ndarray, p2: np.ndarray,
         g1: np.ndarray, g2: np.ndarray) -> float:
    """
    Compute bivariate correlation (cosine similarity in RMM space).

    Args:
        p1, p2: Predictions for RMM1 and RMM2
        g1, g2: Ground truth for RMM1 and RMM2

    Returns:
        Bivariate correlation value
    """
    num = (p1 * g1 + p2 * g2).sum()
    den = np.sqrt((p1**2 + p2**2).sum() * (g1**2 + g2**2).sum())
    return num / (den + 1e-8)


def rmse(p1: np.ndarray, p2: np.ndarray,
         g1: np.ndarray, g2: np.ndarray) -> float:
    """Compute bivariate RMSE."""
    return np.sqrt(np.mean((p1 - g1)**2 + (p2 - g2)**2))


def compute_crps_quantiles(quantiles: np.ndarray, y_true: np.ndarray,
                           quantile_levels: List[float] = None) -> float:
    """
    Compute CRPS from quantile predictions using pinball loss.

    This is the standard method for computing CRPS from quantile predictions.

    Formula: CRPS = (2/K) × Σₖ ρ_τₖ(y - qₖ)
    where ρ_τ(u) = u × (τ - 𝟙(u < 0)) is the pinball/quantile loss.

    References:
        - Gneiting & Raftery (2007): Strictly Proper Scoring Rules
        - Koenker & Bassett (1978): Quantile Regression
        - Laio & Tamea (2007): Verification tools for probabilistic forecasts

    Args:
        quantiles: (n_samples, n_quantiles) predicted quantiles
        y_true: (n_samples,) true values
        quantile_levels: List of quantile levels (default: QUANTILE_LEVELS)

    Returns:
        Average CRPS across samples
    """
    if quantile_levels is None:
        quantile_levels = QUANTILE_LEVELS

    n_samples = len(y_true)
    n_q = len(quantile_levels)

    total_pinball = 0.0

    for k, tau in enumerate(quantile_levels):
        # Error: y - q (positive if y > q, negative if y < q)
        errors = y_true - quantiles[:, k]

        # Pinball loss: ρ_τ(u) = u × (τ - 𝟙(u < 0))
        # If error >= 0 (y >= q): loss = τ × error
        # If error < 0 (y < q): loss = (τ - 1) × error = (1 - τ) × |error|
        pinball_loss = np.where(errors >= 0,
                                tau * errors,
                                (tau - 1) * errors)

        total_pinball += np.mean(pinball_loss)

    # CRPS = 2 × mean pinball loss across quantile levels
    crps = 2 * total_pinball / n_q

    return crps


def compute_coverage(quantiles_rmm1: np.ndarray, quantiles_rmm2: np.ndarray,
                     gt_rmm1: np.ndarray, gt_rmm2: np.ndarray,
                     level: float = 0.9) -> float:
    """
    Compute joint coverage rate.

    Args:
        quantiles_rmm1, quantiles_rmm2: (n_samples, n_quantiles)
        gt_rmm1, gt_rmm2: (n_samples,)
        level: Coverage level (0.9 for 90%)

    Returns:
        Coverage rate (0-1)
    """
    alpha = (1 - level) / 2
    lower_idx = QUANTILE_LEVELS.index(alpha) if alpha in QUANTILE_LEVELS else 0
    upper_idx = QUANTILE_LEVELS.index(1 - alpha) if (1-alpha) in QUANTILE_LEVELS else -1

    covered1 = (gt_rmm1 >= quantiles_rmm1[:, lower_idx]) & (gt_rmm1 <= quantiles_rmm1[:, upper_idx])
    covered2 = (gt_rmm2 >= quantiles_rmm2[:, lower_idx]) & (gt_rmm2 <= quantiles_rmm2[:, upper_idx])

    return (covered1 & covered2).mean()


def compute_spread_skill_ratio(quantiles_rmm1: np.ndarray, quantiles_rmm2: np.ndarray,
                               pred_rmm1: np.ndarray, pred_rmm2: np.ndarray,
                               gt_rmm1: np.ndarray, gt_rmm2: np.ndarray) -> float:
    """
    Compute spread-skill ratio.

    Spread = average prediction uncertainty (from quantiles)
    Skill = actual RMSE
    Ratio should be ~1.0 for well-calibrated predictions.
    """
    # Spread from 90% interval
    q05_idx = QUANTILE_LEVELS.index(0.05)
    q95_idx = QUANTILE_LEVELS.index(0.95)

    spread1 = (quantiles_rmm1[:, q95_idx] - quantiles_rmm1[:, q05_idx]) / (2 * 1.645)
    spread2 = (quantiles_rmm2[:, q95_idx] - quantiles_rmm2[:, q05_idx]) / (2 * 1.645)
    avg_spread = np.mean(np.sqrt((spread1**2 + spread2**2) / 2))

    # Skill (RMSE)
    skill = rmse(pred_rmm1, pred_rmm2, gt_rmm1, gt_rmm2)

    return avg_spread / (skill + 1e-8)


def evaluate_predictions(quantiles_rmm1: np.ndarray, quantiles_rmm2: np.ndarray,
                         gt_rmm1: np.ndarray, gt_rmm2: np.ndarray,
                         baseline_rmm1: np.ndarray = None,
                         baseline_rmm2: np.ndarray = None) -> Dict[str, float]:
    """
    Comprehensive evaluation of probabilistic predictions.

    Args:
        quantiles_rmm1, quantiles_rmm2: (n_samples, n_quantiles)
        gt_rmm1, gt_rmm2: (n_samples,)
        baseline_rmm1, baseline_rmm2: Optional baseline predictions

    Returns:
        Dictionary with all metrics
    """
    median_idx = QUANTILE_LEVELS.index(0.5)
    pred_rmm1 = quantiles_rmm1[:, median_idx]
    pred_rmm2 = quantiles_rmm2[:, median_idx]

    results = {
        'bcor': bcor(pred_rmm1, pred_rmm2, gt_rmm1, gt_rmm2),
        'rmse': rmse(pred_rmm1, pred_rmm2, gt_rmm1, gt_rmm2),
        'crps_rmm1': compute_crps_quantiles(quantiles_rmm1, gt_rmm1),
        'crps_rmm2': compute_crps_quantiles(quantiles_rmm2, gt_rmm2),
        'crps': (compute_crps_quantiles(quantiles_rmm1, gt_rmm1) +
                 compute_crps_quantiles(quantiles_rmm2, gt_rmm2)) / 2,
        'coverage_90': compute_coverage(quantiles_rmm1, quantiles_rmm2, gt_rmm1, gt_rmm2, 0.9),
        'coverage_50': compute_coverage(quantiles_rmm1, quantiles_rmm2, gt_rmm1, gt_rmm2, 0.5),
        'spread_skill_ratio': compute_spread_skill_ratio(
            quantiles_rmm1, quantiles_rmm2, pred_rmm1, pred_rmm2, gt_rmm1, gt_rmm2
        ),
    }

    if baseline_rmm1 is not None and baseline_rmm2 is not None:
        results['bcor_baseline'] = bcor(baseline_rmm1, baseline_rmm2, gt_rmm1, gt_rmm2)
        results['rmse_baseline'] = rmse(baseline_rmm1, baseline_rmm2, gt_rmm1, gt_rmm2)
        results['bcor_improvement'] = (results['bcor'] - results['bcor_baseline']) / results['bcor_baseline'] * 100
        results['rmse_improvement'] = (results['rmse_baseline'] - results['rmse']) / results['rmse_baseline'] * 100

    return results


# ============================================================================
# Loss Functions (PyTorch - for training)
# ============================================================================

def crps_loss_quantile(quantiles: torch.Tensor, targets: torch.Tensor,
                       quantile_levels: torch.Tensor = None) -> torch.Tensor:
    """
    CRPS loss via quantile representation (pinball loss).

    Args:
        quantiles: (batch, n_quantiles) or (batch, 2, n_quantiles) for RMM1/RMM2
        targets: (batch,) or (batch, 2)
        quantile_levels: (n_quantiles,) tensor of quantile levels

    Returns:
        Scalar loss value
    """
    if quantile_levels is None:
        quantile_levels = torch.tensor(QUANTILE_LEVELS, device=quantiles.device)

    # Reshape for broadcasting
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)  # (batch, 1)
    elif targets.dim() == 2 and quantiles.dim() == 3:
        targets = targets.unsqueeze(-1)  # (batch, 2, 1)

    errors = targets - quantiles  # (batch, n_quantiles) or (batch, 2, n_quantiles)

    # Pinball loss
    weights = torch.where(errors >= 0, quantile_levels, quantile_levels - 1)
    pinball = weights * errors

    return pinball.abs().mean()


def crps_loss_gaussian(mu: torch.Tensor, sigma: torch.Tensor,
                       targets: torch.Tensor) -> torch.Tensor:
    """
    CRPS for Gaussian distribution (closed form).

    Args:
        mu: (batch,) or (batch, 2) predicted mean
        sigma: (batch,) or (batch, 2) predicted std (must be positive)
        targets: (batch,) or (batch, 2) true values

    Returns:
        Scalar loss value
    """
    # Standardize
    z = (targets - mu) / (sigma + 1e-8)

    # Standard normal CDF and PDF
    dist = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = dist.cdf(z)
    pdf = torch.exp(dist.log_prob(z))

    # CRPS formula
    crps = sigma * (z * (2 * phi - 1) + 2 * pdf - 1 / np.sqrt(np.pi))

    return crps.mean()


def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for Gaussian distribution.

    Args:
        mu: (batch,) or (batch, 2) predicted mean
        sigma: (batch,) or (batch, 2) predicted std (must be positive)
        targets: (batch,) or (batch, 2) true values

    Returns:
        Scalar loss value
    """
    dist = Normal(mu, sigma + 1e-8)
    return -dist.log_prob(targets).mean()


# ============================================================================
# Quantile Conversion Utilities
# ============================================================================

def gaussian_to_quantiles(mu: torch.Tensor, sigma: torch.Tensor,
                          quantile_levels: List[float] = None) -> torch.Tensor:
    """
    Convert Gaussian parameters to quantiles.

    Args:
        mu: (batch,) or (batch, 2) mean
        sigma: (batch,) or (batch, 2) std
        quantile_levels: List of quantile levels

    Returns:
        quantiles: (batch, n_quantiles) or (batch, 2, n_quantiles)
    """
    if quantile_levels is None:
        quantile_levels = QUANTILE_LEVELS

    dist = Normal(mu.unsqueeze(-1), sigma.unsqueeze(-1) + 1e-8)
    levels = torch.tensor(quantile_levels, device=mu.device, dtype=mu.dtype)
    quantiles = dist.icdf(levels)

    return quantiles


def samples_to_quantiles(samples: torch.Tensor,
                         quantile_levels: List[float] = None) -> torch.Tensor:
    """
    Convert samples to quantiles.

    Args:
        samples: (batch, n_samples) or (batch, 2, n_samples)
        quantile_levels: List of quantile levels

    Returns:
        quantiles: (batch, n_quantiles) or (batch, 2, n_quantiles)
    """
    if quantile_levels is None:
        quantile_levels = QUANTILE_LEVELS

    levels = torch.tensor(quantile_levels, device=samples.device, dtype=samples.dtype)
    quantiles = torch.quantile(samples, levels, dim=-1).permute(1, 0) if samples.dim() == 2 else \
                torch.quantile(samples, levels, dim=-1).permute(1, 2, 0)

    return quantiles


# ============================================================================
# Dataset Classes
# ============================================================================

class BOMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for BOM MJO data."""

    def __init__(self, data: Dict[str, np.ndarray], flatten_leads: bool = True):
        """
        Args:
            data: Dictionary from load_bom_data or split_data_by_years
            flatten_leads: If True, flatten (samples, leads) to single dimension
        """
        self.flatten_leads = flatten_leads

        # Store as tensors
        self.ensemble_rmm1 = torch.tensor(data['ensemble_rmm1'], dtype=torch.float32)
        self.ensemble_rmm2 = torch.tensor(data['ensemble_rmm2'], dtype=torch.float32)
        self.mean_rmm1 = torch.tensor(data['mean_rmm1'], dtype=torch.float32)
        self.mean_rmm2 = torch.tensor(data['mean_rmm2'], dtype=torch.float32)
        self.std_rmm1 = torch.tensor(data['std_rmm1'], dtype=torch.float32)
        self.std_rmm2 = torch.tensor(data['std_rmm2'], dtype=torch.float32)
        self.gt_rmm1 = torch.tensor(data['gt_rmm1'], dtype=torch.float32)
        self.gt_rmm2 = torch.tensor(data['gt_rmm2'], dtype=torch.float32)

        if flatten_leads:
            # Flatten (n_samples, n_leads) -> (n_samples * n_leads,)
            n_samples, n_leads = self.mean_rmm1.shape

            # For ensemble: (n_samples, n_leads, n_members) -> (n_samples * n_leads, n_members)
            self.ensemble_rmm1 = self.ensemble_rmm1.reshape(-1, self.ensemble_rmm1.shape[-1])
            self.ensemble_rmm2 = self.ensemble_rmm2.reshape(-1, self.ensemble_rmm2.shape[-1])

            self.mean_rmm1 = self.mean_rmm1.reshape(-1)
            self.mean_rmm2 = self.mean_rmm2.reshape(-1)
            self.std_rmm1 = self.std_rmm1.reshape(-1)
            self.std_rmm2 = self.std_rmm2.reshape(-1)
            self.gt_rmm1 = self.gt_rmm1.reshape(-1)
            self.gt_rmm2 = self.gt_rmm2.reshape(-1)

            # Create lead day indices
            self.lead_days = torch.arange(1, n_leads + 1).repeat(n_samples)

    def __len__(self):
        return len(self.mean_rmm1)

    def __getitem__(self, idx):
        return {
            'ensemble_rmm1': self.ensemble_rmm1[idx],
            'ensemble_rmm2': self.ensemble_rmm2[idx],
            'mean_rmm1': self.mean_rmm1[idx],
            'mean_rmm2': self.mean_rmm2[idx],
            'std_rmm1': self.std_rmm1[idx],
            'std_rmm2': self.std_rmm2[idx],
            'gt_rmm1': self.gt_rmm1[idx],
            'gt_rmm2': self.gt_rmm2[idx],
            'lead_day': self.lead_days[idx] if hasattr(self, 'lead_days') else 0,
        }


# ============================================================================
# Training Utilities
# ============================================================================

def save_model(model: nn.Module, path: str, config: Dict = None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config or {},
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return checkpoint.get('config', {})


def get_device():
    """Get available device (GPU if available)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device
