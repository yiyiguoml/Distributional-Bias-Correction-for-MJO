#!/usr/bin/env python3
"""
EMOS (Ensemble Model Output Statistics) / NGR (Non-homogeneous Gaussian Regression)
with BIVARIATE Gaussian for joint RMM1-RMM2 prediction.

Reference:
    Gneiting et al. (2005): Calibrated Probabilistic Forecasting Using Ensemble
    Model Output Statistics and Minimum CRPS Estimation

Method:
    Fit a BIVARIATE Gaussian distribution where mean, variance, and correlation
    are functions of ensemble statistics:
        μ₁ = a₁ + b₁ * ensemble_mean_rmm1
        μ₂ = a₂ + b₂ * ensemble_mean_rmm2
        σ₁² = exp(c₁ + d₁ * ensemble_spread_rmm1)
        σ₂² = exp(c₂ + d₂ * ensemble_spread_rmm2)
        ρ = tanh(ρ_raw)  # correlation coefficient ∈ (-1, 1)

    Trained by minimizing Energy Score (multivariate CRPS) over the training set.

Input: Ensemble mean + ensemble spread (2 features per RMM component)
Output: Bivariate Gaussian parameters (μ₁, μ₂, σ₁, σ₂, ρ) converted to quantiles
Model scope: Separate model per lead day (62 models)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import json
import argparse
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_dataset, DATASET_CONFIGS, QUANTILE_LEVELS,
    evaluate_predictions, save_model, get_device, set_seed, SEED
)

# Set seed for reproducibility
set_seed(SEED)

# ============================================================================
# Hyperparameters
# ============================================================================

HYPERPARAMS = {
    'learning_rate': 0.001,    # Standardized across all baselines
    'max_epochs': 500,         # Standardized across all baselines
    'patience': 20,            # Standardized across all baselines
    'batch_size': 256,
    'weight_decay': 0.0,
    'n_samples': 100,  # Number of samples for Energy Score
}

# ============================================================================
# Model Definition
# ============================================================================

class BivariateEMOS(nn.Module):
    """
    EMOS with bivariate Gaussian for joint RMM1-RMM2 prediction.

    Parameters:
        a: (2,) intercepts for means μ₁, μ₂
        b: (2,) slopes for means (on ens_mean)
        e: (2,) slopes for means (on amplitude)
        c: (2,) intercepts for log variance
        d: (2,) slopes for log variance (on ens_spread)
        f: (2,) slopes for log variance (on amplitude_std)
        rho_raw: (1,) raw correlation parameter (tanh → ρ)

    Total: 13 learnable parameters per lead day (was 9 without amplitude)
    """

    def __init__(self):
        super().__init__()
        # Mean parameters: μ = a + b * ens_mean + e * amplitude_mean
        self.a = nn.Parameter(torch.zeros(2))  # intercepts
        self.b = nn.Parameter(torch.ones(2))   # slopes for ens_mean
        self.e = nn.Parameter(torch.zeros(2))  # slopes for amplitude (initialized to 0)

        # Variance parameters: log(σ²) = c + d * ens_spread + f * amplitude_std
        self.c = nn.Parameter(torch.zeros(2))  # intercepts
        self.d = nn.Parameter(torch.ones(2))   # slopes for ens_spread
        self.f = nn.Parameter(torch.zeros(2))  # slopes for amplitude_std (initialized to 0)

        # Correlation parameter: ρ = tanh(rho_raw)
        self.rho_raw = nn.Parameter(torch.zeros(1))

    def forward(self, ens_mean: torch.Tensor, ens_spread: torch.Tensor,
                amplitude_mean: torch.Tensor = None, amplitude_std: torch.Tensor = None):
        """
        Forward pass.

        Args:
            ens_mean: (batch, 2) ensemble mean for RMM1, RMM2
            ens_spread: (batch, 2) ensemble spread for RMM1, RMM2
            amplitude_mean: (batch,) ensemble mean amplitude (optional for backward compat)
            amplitude_std: (batch,) ensemble std amplitude (optional for backward compat)

        Returns:
            mu: (batch, 2) predicted means
            sigma: (batch, 2) predicted stds
            rho: scalar correlation coefficient
        """
        mu = self.a + self.b * ens_mean
        log_var = self.c + self.d * ens_spread

        # Add amplitude contribution if provided
        if amplitude_mean is not None:
            amp_mean = amplitude_mean.unsqueeze(-1) if amplitude_mean.dim() == 1 else amplitude_mean
            mu = mu + self.e * amp_mean
        if amplitude_std is not None:
            amp_std = amplitude_std.unsqueeze(-1) if amplitude_std.dim() == 1 else amplitude_std
            log_var = log_var + self.f * amp_std

        sigma = torch.exp(0.5 * log_var)
        sigma = torch.clamp(sigma, min=0.01, max=10.0)
        rho = torch.tanh(self.rho_raw)

        return mu, sigma, rho

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor, rho: torch.Tensor,
               n_samples: int = 100) -> torch.Tensor:
        """
        Sample from bivariate Gaussian using Cholesky decomposition.

        Returns:
            samples: (batch, n_samples, 2)
        """
        batch_size = mu.shape[0]
        device = mu.device

        # Cholesky decomposition of covariance matrix
        # [[σ₁², ρσ₁σ₂], [ρσ₁σ₂, σ₂²]]
        # L = [[L11, 0], [L21, L22]]
        L11 = sigma[:, 0]  # (batch,)
        L21 = rho * sigma[:, 1]  # (batch,)
        L22 = sigma[:, 1] * torch.sqrt(1 - rho**2 + 1e-8)  # (batch,)

        # Standard normal samples
        z = torch.randn(batch_size, n_samples, 2, device=device)

        # Transform: x = μ + L @ z
        samples = torch.zeros(batch_size, n_samples, 2, device=device)
        samples[:, :, 0] = mu[:, 0:1] + L11.unsqueeze(1) * z[:, :, 0]
        samples[:, :, 1] = mu[:, 1:2] + L21.unsqueeze(1) * z[:, :, 0] + L22.unsqueeze(1) * z[:, :, 1]

        return samples

    def predict_quantiles(self, ens_mean: torch.Tensor, ens_spread: torch.Tensor,
                          amplitude_mean: torch.Tensor = None, amplitude_std: torch.Tensor = None,
                          n_samples: int = 1000) -> torch.Tensor:
        """
        Predict quantiles by sampling from bivariate Gaussian.

        Returns:
            quantiles: (batch, 2, n_quantiles)
        """
        mu, sigma, rho = self.forward(ens_mean, ens_spread, amplitude_mean, amplitude_std)
        samples = self.sample(mu, sigma, rho, n_samples)  # (batch, n_samples, 2)

        # Compute quantiles from samples
        levels = torch.tensor(QUANTILE_LEVELS, device=samples.device)
        quantiles_rmm1 = torch.quantile(samples[:, :, 0], levels, dim=1).T  # (batch, n_quantiles)
        quantiles_rmm2 = torch.quantile(samples[:, :, 1], levels, dim=1).T

        return torch.stack([quantiles_rmm1, quantiles_rmm2], dim=1)  # (batch, 2, n_quantiles)


# ============================================================================
# Loss Functions
# ============================================================================

def energy_score(samples: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Energy Score (multivariate CRPS) for bivariate predictions.

    ES = E||X - y|| - 0.5 * E||X - X'||

    Args:
        samples: (batch, n_samples, 2) samples from predictive distribution
        targets: (batch, 2) true RMM1, RMM2 values

    Returns:
        Scalar energy score (averaged over batch)
    """
    batch_size, n_samples, _ = samples.shape

    # Term 1: E||X - y||
    diff_to_target = samples - targets.unsqueeze(1)  # (batch, n_samples, 2)
    term1 = torch.norm(diff_to_target, dim=-1).mean(dim=1)  # (batch,)

    # Term 2: 0.5 * E||X - X'||
    # For efficiency, use pairwise distances within samples
    # Subsampling for large n_samples
    if n_samples > 50:
        idx = torch.randperm(n_samples)[:50]
        samples_sub = samples[:, idx, :]
        n_sub = 50
    else:
        samples_sub = samples
        n_sub = n_samples

    # (batch, n_sub, 1, 2) - (batch, 1, n_sub, 2) = (batch, n_sub, n_sub, 2)
    diff_between_samples = samples_sub.unsqueeze(2) - samples_sub.unsqueeze(1)
    pairwise_distances = torch.norm(diff_between_samples, dim=-1)  # (batch, n_sub, n_sub)
    term2 = 0.5 * pairwise_distances.mean(dim=(1, 2))  # (batch,)

    return (term1 - term2).mean()


# ============================================================================
# Training Functions
# ============================================================================

def prepare_data_for_lead(data: dict, lead_day: int):
    """Extract data for a specific lead day."""
    ens_mean = np.stack([
        data['mean_rmm1'][:, lead_day],
        data['mean_rmm2'][:, lead_day]
    ], axis=1)

    ens_spread = np.stack([
        data['std_rmm1'][:, lead_day],
        data['std_rmm2'][:, lead_day]
    ], axis=1)

    targets = np.stack([
        data['gt_rmm1'][:, lead_day],
        data['gt_rmm2'][:, lead_day]
    ], axis=1)

    # Amplitude data
    amplitude_mean = data['amplitude_mean'][:, lead_day]
    amplitude_std = data['amplitude_std'][:, lead_day]

    # Remove NaN samples
    valid = ~(np.isnan(ens_mean).any(axis=1) |
              np.isnan(ens_spread).any(axis=1) |
              np.isnan(targets).any(axis=1) |
              np.isnan(amplitude_mean) |
              np.isnan(amplitude_std))

    return (ens_mean[valid], ens_spread[valid], amplitude_mean[valid],
            amplitude_std[valid], targets[valid])


def train_emos_single_lead(train_data: dict, valid_data: dict,
                           lead_day: int, device: torch.device,
                           hyperparams: dict = None) -> tuple:
    """Train bivariate EMOS model for a single lead day."""
    if hyperparams is None:
        hyperparams = HYPERPARAMS

    # Prepare data (now includes amplitude)
    (train_mean, train_spread, train_amp_mean,
     train_amp_std, train_targets) = prepare_data_for_lead(train_data, lead_day)
    (valid_mean, valid_spread, valid_amp_mean,
     valid_amp_std, valid_targets) = prepare_data_for_lead(valid_data, lead_day)

    # Normalize inputs (stack all features for consistent scaling)
    train_features = np.hstack([train_mean, train_spread,
                                train_amp_mean.reshape(-1, 1),
                                train_amp_std.reshape(-1, 1)])
    valid_features = np.hstack([valid_mean, valid_spread,
                                valid_amp_mean.reshape(-1, 1),
                                valid_amp_std.reshape(-1, 1)])

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    valid_features = scaler.transform(valid_features)

    # Unstack back
    train_mean = train_features[:, :2]
    train_spread = train_features[:, 2:4]
    train_amp_mean = train_features[:, 4]
    train_amp_std = train_features[:, 5]
    valid_mean = valid_features[:, :2]
    valid_spread = valid_features[:, 2:4]
    valid_amp_mean = valid_features[:, 4]
    valid_amp_std = valid_features[:, 5]

    # Convert to tensors
    train_mean = torch.tensor(train_mean, dtype=torch.float32, device=device)
    train_spread = torch.tensor(train_spread, dtype=torch.float32, device=device)
    train_amp_mean = torch.tensor(train_amp_mean, dtype=torch.float32, device=device)
    train_amp_std = torch.tensor(train_amp_std, dtype=torch.float32, device=device)
    train_targets = torch.tensor(train_targets, dtype=torch.float32, device=device)

    valid_mean = torch.tensor(valid_mean, dtype=torch.float32, device=device)
    valid_spread = torch.tensor(valid_spread, dtype=torch.float32, device=device)
    valid_amp_mean = torch.tensor(valid_amp_mean, dtype=torch.float32, device=device)
    valid_amp_std = torch.tensor(valid_amp_std, dtype=torch.float32, device=device)
    valid_targets = torch.tensor(valid_targets, dtype=torch.float32, device=device)

    # Create data loader (now with 5 tensors)
    train_dataset = TensorDataset(train_mean, train_spread, train_amp_mean,
                                   train_amp_std, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)

    # Initialize model
    model = BivariateEMOS().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=hyperparams['learning_rate'],
                           weight_decay=hyperparams['weight_decay'])

    # Training loop
    history = {'train_loss': [], 'valid_loss': []}
    best_valid_loss = float('inf')
    best_state = None
    patience_counter = 0
    n_samples = hyperparams['n_samples']

    for epoch in range(hyperparams['max_epochs']):
        # Training
        model.train()
        train_losses = []
        for batch_mean, batch_spread, batch_amp_mean, batch_amp_std, batch_targets in train_loader:
            optimizer.zero_grad()
            mu, sigma, rho = model(batch_mean, batch_spread, batch_amp_mean, batch_amp_std)
            samples = model.sample(mu, sigma, rho, n_samples)
            loss = energy_score(samples, batch_targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            mu, sigma, rho = model(valid_mean, valid_spread, valid_amp_mean, valid_amp_std)
            samples = model.sample(mu, sigma, rho, n_samples)
            valid_loss = energy_score(samples, valid_targets).item()

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(valid_loss)

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= hyperparams['patience']:
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, history


def train_emos_all_leads(train_data: dict, valid_data: dict,
                         device: torch.device,
                         hyperparams: dict = None,
                         n_lead_days: int = 62) -> dict:
    """Train EMOS models for all lead days."""
    models = {}
    scalers = {}

    print(f"Training Bivariate EMOS for {n_lead_days} lead days...")

    for lead in range(n_lead_days):
        print(f"  Lead day {lead+1}/{n_lead_days}...", end=' ', flush=True)
        model, scaler, history = train_emos_single_lead(
            train_data, valid_data, lead, device, hyperparams
        )
        models[lead] = model
        scalers[lead] = scaler
        print(f"Valid ES: {history['valid_loss'][-1]:.4f} (epochs: {len(history['valid_loss'])})")

    return models, scalers


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_emos(models: dict, scalers: dict, test_data: dict, device: torch.device,
                  n_lead_days: int = 62) -> dict:
    """Evaluate EMOS models on test data."""
    n_samples_test = test_data['mean_rmm1'].shape[0]
    n_quantiles = len(QUANTILE_LEVELS)

    all_quantiles_rmm1 = np.zeros((n_samples_test, n_lead_days, n_quantiles))
    all_quantiles_rmm2 = np.zeros((n_samples_test, n_lead_days, n_quantiles))

    for lead in range(n_lead_days):
        model = models[lead]
        scaler = scalers[lead]
        model.eval()

        ens_mean = np.stack([
            test_data['mean_rmm1'][:, lead],
            test_data['mean_rmm2'][:, lead]
        ], axis=1)
        ens_spread = np.stack([
            test_data['std_rmm1'][:, lead],
            test_data['std_rmm2'][:, lead]
        ], axis=1)

        # Amplitude data
        amp_mean = test_data['amplitude_mean'][:, lead]
        amp_std = test_data['amplitude_std'][:, lead]

        # Apply same normalization used during training
        test_features = np.hstack([ens_mean, ens_spread,
                                   amp_mean.reshape(-1, 1),
                                   amp_std.reshape(-1, 1)])
        test_features = scaler.transform(test_features)

        # Unstack back
        ens_mean = test_features[:, :2]
        ens_spread = test_features[:, 2:4]
        amp_mean = test_features[:, 4]
        amp_std = test_features[:, 5]

        ens_mean_t = torch.tensor(ens_mean, dtype=torch.float32, device=device)
        ens_spread_t = torch.tensor(ens_spread, dtype=torch.float32, device=device)
        amp_mean_t = torch.tensor(amp_mean, dtype=torch.float32, device=device)
        amp_std_t = torch.tensor(amp_std, dtype=torch.float32, device=device)

        with torch.no_grad():
            quantiles = model.predict_quantiles(ens_mean_t, ens_spread_t,
                                                 amp_mean_t, amp_std_t, n_samples=1000)
            quantiles = quantiles.cpu().numpy()

        all_quantiles_rmm1[:, lead, :] = quantiles[:, 0, :]
        all_quantiles_rmm2[:, lead, :] = quantiles[:, 1, :]

    # Compute metrics per lead day
    results_per_lead = []
    for lead in range(n_lead_days):
        q1 = all_quantiles_rmm1[:, lead, :]
        q2 = all_quantiles_rmm2[:, lead, :]
        g1 = test_data['gt_rmm1'][:, lead]
        g2 = test_data['gt_rmm2'][:, lead]
        b1 = test_data['mean_rmm1'][:, lead]
        b2 = test_data['mean_rmm2'][:, lead]

        valid = ~(np.isnan(g1) | np.isnan(g2))
        if valid.sum() == 0:
            continue

        metrics = evaluate_predictions(
            q1[valid], q2[valid], g1[valid], g2[valid], b1[valid], b2[valid]
        )
        metrics['lead_day'] = lead + 1
        results_per_lead.append(metrics)

    results = {
        'per_lead': results_per_lead,
        'mean_crps': np.mean([r['crps'] for r in results_per_lead]),
        'mean_bcor': np.mean([r['bcor'] for r in results_per_lead]),
        'mean_rmse': np.mean([r['rmse'] for r in results_per_lead]),
        'mean_coverage_90': np.mean([r['coverage_90'] for r in results_per_lead]),
        'mean_spread_skill': np.mean([r['spread_skill_ratio'] for r in results_per_lead]),
    }

    return results


# ============================================================================
# Walk-Forward Validation
# ============================================================================

def run_walk_forward(save_dir: str, dataset_name: str = 'BoM', hyperparams: dict = None):
    """Run walk-forward validation for bivariate EMOS."""
    if hyperparams is None:
        hyperparams = HYPERPARAMS

    # Reset seed for each run
    set_seed(SEED)

    device = get_device()
    config = DATASET_CONFIGS[dataset_name]
    test_years = config['test_years']
    n_lead_days = config['n_lead_days']
    train_window = config['train_window']

    print(f"Loading {dataset_name} data...")
    all_data = load_dataset(dataset_name)

    # Get year indices from times
    import pandas as pd
    times = pd.to_datetime(all_data['times'])
    years_array = times.year.values

    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)

    all_results = []

    for test_year in test_years:
        set_seed(SEED)  # Reset seed for each fold

        print(f"\n{'='*60}")
        print(f"Test Year: {test_year}")
        print(f"{'='*60}")

        # Walk-forward: validation is always year before test, training is everything before that
        # This ensures NO data leakage between train/valid/test
        valid_year = test_year - 1
        train_end_year = valid_year - 1
        train_start_year = train_end_year - train_window + 2  # +2 because we took 1 year for validation
        train_years = list(range(train_start_year, train_end_year + 1))

        if len(train_years) < 2:
            print(f"  Skipping {test_year}: not enough training data")
            continue

        print(f"  Train years: {train_years[0]}-{train_years[-1]} ({len(train_years)} years)")
        print(f"  Valid year: {valid_year}")

        def get_data_for_years(years):
            mask = np.isin(years_array, years)
            return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == len(years_array) else v
                    for k, v in all_data.items()}

        train_data = get_data_for_years(train_years)
        valid_data = get_data_for_years([valid_year])
        test_data = get_data_for_years([test_year])

        print(f"  Train samples: {train_data['mean_rmm1'].shape[0]}")
        print(f"  Valid samples: {valid_data['mean_rmm1'].shape[0]}")
        print(f"  Test samples: {test_data['mean_rmm1'].shape[0]}")

        if train_data['mean_rmm1'].shape[0] == 0 or test_data['mean_rmm1'].shape[0] == 0:
            print(f"  Skipping {test_year}: no data")
            continue

        models, scalers = train_emos_all_leads(train_data, valid_data, device, hyperparams, n_lead_days)

        model_path = os.path.join(save_dir, 'models', f'emos_fold_{test_year}.pt')
        torch.save({
            'models': {lead: model.state_dict() for lead, model in models.items()},
            'scalers': scalers,  # Save scalers for inference
            'hyperparams': hyperparams,
            'test_year': test_year,
            'dataset': dataset_name,
        }, model_path)
        print(f"  Models saved to {model_path}")

        results = evaluate_emos(models, scalers, test_data, device, n_lead_days)
        results['test_year'] = test_year

        print(f"\n  Results for {test_year}:")
        print(f"    CRPS: {results['mean_crps']:.4f}")
        print(f"    BCOR: {results['mean_bcor']:.4f}")
        print(f"    RMSE: {results['mean_rmse']:.4f}")
        print(f"    Coverage (90%): {results['mean_coverage_90']*100:.1f}%")
        print(f"    Spread-Skill: {results['mean_spread_skill']:.2f}")

        all_results.append({
            'test_year': test_year,
            'crps': results['mean_crps'],
            'bcor': results['mean_bcor'],
            'rmse': results['mean_rmse'],
            'coverage_90': results['mean_coverage_90'],
            'spread_skill': results['mean_spread_skill'],
        })

    summary = {
        'method': 'EMOS (Bivariate)',
        'dataset': dataset_name,
        'seed': SEED,
        'hyperparams': hyperparams,
        'results_per_year': all_results,
        'mean_crps': np.mean([r['crps'] for r in all_results]) if all_results else 0,
        'mean_bcor': np.mean([r['bcor'] for r in all_results]) if all_results else 0,
        'mean_rmse': np.mean([r['rmse'] for r in all_results]) if all_results else 0,
        'mean_coverage_90': np.mean([r['coverage_90'] for r in all_results]) if all_results else 0,
        'mean_spread_skill': np.mean([r['spread_skill'] for r in all_results]) if all_results else 0,
        'std_crps': np.std([r['crps'] for r in all_results]) if all_results else 0,
        'std_bcor': np.std([r['bcor'] for r in all_results]) if all_results else 0,
        'std_rmse': np.std([r['rmse'] for r in all_results]) if all_results else 0,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(save_dir, 'results', 'emos_summary.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"EMOS (Bivariate) - {dataset_name} Final Summary")
    print(f"{'='*60}")
    print(f"CRPS: {summary['mean_crps']:.4f} +/- {summary['std_crps']:.4f}")
    print(f"BCOR: {summary['mean_bcor']:.4f} +/- {summary['std_bcor']:.4f}")
    print(f"RMSE: {summary['mean_rmse']:.4f} +/- {summary['std_rmse']:.4f}")
    print(f"Coverage (90%): {summary['mean_coverage_90']*100:.1f}%")
    print(f"Spread-Skill: {summary['mean_spread_skill']:.2f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Train Bivariate EMOS baseline')
    parser.add_argument('--dataset', type=str, default='BoM',
                        choices=['BoM', 'JMA', 'CNRM'],
                        help='Dataset to use')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results (default: auto based on dataset)')
    parser.add_argument('--lr', type=float, default=HYPERPARAMS['learning_rate'])
    parser.add_argument('--epochs', type=int, default=HYPERPARAMS['max_epochs'])
    parser.add_argument('--patience', type=int, default=HYPERPARAMS['patience'])

    args = parser.parse_args()

    # Set save directory based on dataset if not specified
    if args.save_dir is None:
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(_current_dir)  # baselines/ directory
        args.save_dir = os.path.join(base_dir, 'probabilistic', args.dataset)

    os.makedirs(args.save_dir, exist_ok=True)

    hyperparams = HYPERPARAMS.copy()
    hyperparams['learning_rate'] = args.lr
    hyperparams['max_epochs'] = args.epochs
    hyperparams['patience'] = args.patience

    run_walk_forward(args.save_dir, args.dataset, hyperparams)


if __name__ == '__main__':
    main()
