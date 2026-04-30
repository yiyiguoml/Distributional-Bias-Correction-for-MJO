#!/usr/bin/env python3
"""
BMA (Bayesian Model Averaging) with Bivariate Gaussian Mixture

Reference:
    Raftery et al. (2005): Using Bayesian Model Averaging to Calibrate Forecast Ensembles

Method:
    Mixture of bivariate Gaussians centered on each ensemble member:
    p(y) = Σₖ wₖ * N(y | xₖ, Σ)

    where:
    - wₖ: learnable weights for each member (softmax to sum to 1)
    - xₖ: (RMM1, RMM2) from ensemble member k
    - Σ: shared covariance matrix with correlation parameter ρ

    Trained by minimizing Energy Score (multivariate CRPS).

Input: All 33 ensemble members
Output: Samples from mixture → quantiles at [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
Model scope: Separate model per lead day (62 models)

NOTE: Uses all 33 ensemble members as input - MORE information than DBC.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    load_dataset_with_ensemble, DATASET_CONFIGS, QUANTILE_LEVELS,
    evaluate_predictions, get_device, set_seed, SEED
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
    'n_samples': 100,
}

# ============================================================================
# Model Definition
# ============================================================================

class BivariateBMA(nn.Module):
    """
    Bayesian Model Averaging with bivariate Gaussian mixture.

    Parameters:
        raw_weights: (33,) unnormalized log-weights
        log_sigma: (2,) log standard deviations for σ₁, σ₂
        amp_sigma_scale: (2,) scale for amplitude contribution to log variance
        rho_raw: (1,) raw correlation parameter

    Total: 38 learnable parameters per lead day (was 36 without amplitude)
    """

    def __init__(self, n_members: int = 33):
        super().__init__()
        self.n_members = n_members

        # Learnable weights (softmax to sum to 1)
        self.raw_weights = nn.Parameter(torch.zeros(n_members))

        # Shared variance parameters: log(σ²) = log_sigma_base + amp_sigma_scale * amplitude_std
        self.log_sigma = nn.Parameter(torch.zeros(2))  # log(σ₁), log(σ₂)
        self.amp_sigma_scale = nn.Parameter(torch.zeros(2))  # amplitude contribution (init to 0)

        # Correlation parameter
        self.rho_raw = nn.Parameter(torch.zeros(1))

    def forward(self, ensemble_rmm1: torch.Tensor, ensemble_rmm2: torch.Tensor,
                amplitude_std: torch.Tensor = None):
        """
        Forward pass - compute mixture parameters.

        Args:
            ensemble_rmm1: (batch, 33) all members for RMM1
            ensemble_rmm2: (batch, 33) all members for RMM2
            amplitude_std: (batch,) ensemble std amplitude (optional)

        Returns:
            weights: (33,) normalized weights
            sigma: (batch, 2) or (2,) standard deviations
            rho: scalar correlation
        """
        weights = F.softmax(self.raw_weights, dim=0)

        # Base log variance
        log_var = self.log_sigma

        # Add amplitude contribution if provided
        if amplitude_std is not None:
            amp_std = amplitude_std.unsqueeze(-1) if amplitude_std.dim() == 1 else amplitude_std
            # Broadcast: (batch, 1) * (2,) -> (batch, 2)
            log_var = log_var + self.amp_sigma_scale * amp_std

        sigma = torch.exp(0.5 * log_var)  # sqrt(exp(log_var)) = exp(0.5 * log_var)
        sigma = torch.clamp(sigma, min=0.01, max=5.0)
        rho = torch.tanh(self.rho_raw)

        return weights, sigma, rho

    def sample(self, ensemble_rmm1: torch.Tensor, ensemble_rmm2: torch.Tensor,
               weights: torch.Tensor, sigma: torch.Tensor, rho: torch.Tensor,
               n_samples: int = 100) -> torch.Tensor:
        """
        Sample from mixture of bivariate Gaussians.

        Args:
            sigma: (2,) or (batch, 2) standard deviations

        Returns:
            samples: (batch, n_samples, 2)
        """
        batch_size = ensemble_rmm1.shape[0]
        device = ensemble_rmm1.device

        # Sample which component (member) for each sample
        component_idx = torch.multinomial(
            weights.expand(batch_size, -1),
            n_samples, replacement=True
        )  # (batch, n_samples)

        # Get member values for selected components
        member_rmm1 = torch.gather(ensemble_rmm1, 1, component_idx)  # (batch, n_samples)
        member_rmm2 = torch.gather(ensemble_rmm2, 1, component_idx)

        # Handle batch-specific sigma
        if sigma.dim() == 1:
            # sigma is (2,) - same for all samples
            sigma1 = sigma[0]
            sigma2 = sigma[1]
        else:
            # sigma is (batch, 2) - different per sample
            sigma1 = sigma[:, 0:1]  # (batch, 1)
            sigma2 = sigma[:, 1:2]  # (batch, 1)

        # Sample from bivariate Gaussian centered on member
        # Cholesky decomposition
        L11 = sigma1
        L21 = rho * sigma2
        L22 = sigma2 * torch.sqrt(1 - rho**2 + 1e-8)

        z = torch.randn(batch_size, n_samples, 2, device=device)
        samples = torch.zeros(batch_size, n_samples, 2, device=device)

        samples[:, :, 0] = member_rmm1 + L11 * z[:, :, 0]
        samples[:, :, 1] = member_rmm2 + L21 * z[:, :, 0] + L22 * z[:, :, 1]

        return samples

    def predict_quantiles(self, ensemble_rmm1: torch.Tensor, ensemble_rmm2: torch.Tensor,
                          amplitude_std: torch.Tensor = None,
                          n_samples: int = 1000) -> torch.Tensor:
        """
        Predict quantiles by sampling from mixture.

        Returns:
            quantiles: (batch, 2, n_quantiles)
        """
        weights, sigma, rho = self.forward(ensemble_rmm1, ensemble_rmm2, amplitude_std)
        samples = self.sample(ensemble_rmm1, ensemble_rmm2, weights, sigma, rho, n_samples)

        levels = torch.tensor(QUANTILE_LEVELS, device=samples.device)
        quantiles_rmm1 = torch.quantile(samples[:, :, 0], levels, dim=1).T
        quantiles_rmm2 = torch.quantile(samples[:, :, 1], levels, dim=1).T

        return torch.stack([quantiles_rmm1, quantiles_rmm2], dim=1)


# ============================================================================
# Loss Functions
# ============================================================================

def energy_score(samples: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Energy Score (multivariate CRPS)."""
    batch_size, n_samples, _ = samples.shape

    # Term 1: E||X - y||
    diff_to_target = samples - targets.unsqueeze(1)
    term1 = torch.norm(diff_to_target, dim=-1).mean(dim=1)

    # Term 2: 0.5 * E||X - X'||
    if n_samples > 50:
        idx = torch.randperm(n_samples)[:50]
        samples_sub = samples[:, idx, :]
    else:
        samples_sub = samples

    diff_between_samples = samples_sub.unsqueeze(2) - samples_sub.unsqueeze(1)
    pairwise_distances = torch.norm(diff_between_samples, dim=-1)
    term2 = 0.5 * pairwise_distances.mean(dim=(1, 2))

    return (term1 - term2).mean()


# ============================================================================
# Training Functions
# ============================================================================

def prepare_data_for_lead(data: dict, lead_day: int):
    """Extract data for a specific lead day."""
    ensemble_rmm1 = data['ensemble_rmm1'][:, lead_day, :]  # (n_samples, 33)
    ensemble_rmm2 = data['ensemble_rmm2'][:, lead_day, :]

    targets = np.stack([
        data['gt_rmm1'][:, lead_day],
        data['gt_rmm2'][:, lead_day]
    ], axis=1)

    # Amplitude data
    amplitude_std = data['amplitude_std'][:, lead_day]

    # Remove NaN samples
    valid = ~(np.isnan(ensemble_rmm1).any(axis=1) |
              np.isnan(ensemble_rmm2).any(axis=1) |
              np.isnan(targets).any(axis=1) |
              np.isnan(amplitude_std))

    return ensemble_rmm1[valid], ensemble_rmm2[valid], amplitude_std[valid], targets[valid]


def train_bma_single_lead(train_data: dict, valid_data: dict,
                          lead_day: int, device: torch.device,
                          hyperparams: dict = None, n_members: int = 33) -> tuple:
    """Train BMA model for a single lead day."""
    if hyperparams is None:
        hyperparams = HYPERPARAMS

    (train_ens1, train_ens2, train_amp_std,
     train_targets) = prepare_data_for_lead(train_data, lead_day)
    (valid_ens1, valid_ens2, valid_amp_std,
     valid_targets) = prepare_data_for_lead(valid_data, lead_day)

    # Normalize ensemble members (stack all 66 features for consistent scaling)
    train_features = np.hstack([train_ens1, train_ens2, train_amp_std.reshape(-1, 1)])
    valid_features = np.hstack([valid_ens1, valid_ens2, valid_amp_std.reshape(-1, 1)])

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    valid_features = scaler.transform(valid_features)

    # Unstack back
    train_ens1 = train_features[:, :n_members]
    train_ens2 = train_features[:, n_members:2*n_members]
    train_amp_std = train_features[:, -1]
    valid_ens1 = valid_features[:, :n_members]
    valid_ens2 = valid_features[:, n_members:2*n_members]
    valid_amp_std = valid_features[:, -1]

    # Convert to tensors
    train_ens1 = torch.tensor(train_ens1, dtype=torch.float32, device=device)
    train_ens2 = torch.tensor(train_ens2, dtype=torch.float32, device=device)
    train_amp_std = torch.tensor(train_amp_std, dtype=torch.float32, device=device)
    train_targets = torch.tensor(train_targets, dtype=torch.float32, device=device)

    valid_ens1 = torch.tensor(valid_ens1, dtype=torch.float32, device=device)
    valid_ens2 = torch.tensor(valid_ens2, dtype=torch.float32, device=device)
    valid_amp_std = torch.tensor(valid_amp_std, dtype=torch.float32, device=device)
    valid_targets = torch.tensor(valid_targets, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(train_ens1, train_ens2, train_amp_std, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)

    model = BivariateBMA(n_members=n_members).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'],
                           weight_decay=hyperparams['weight_decay'])

    history = {'train_loss': [], 'valid_loss': []}
    best_valid_loss = float('inf')
    best_state = None
    patience_counter = 0
    n_samples = hyperparams['n_samples']

    for epoch in range(hyperparams['max_epochs']):
        model.train()
        train_losses = []
        for batch_ens1, batch_ens2, batch_amp_std, batch_targets in train_loader:
            optimizer.zero_grad()
            weights, sigma, rho = model(batch_ens1, batch_ens2, batch_amp_std)
            samples = model.sample(batch_ens1, batch_ens2, weights, sigma, rho, n_samples)
            loss = energy_score(samples, batch_targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            weights, sigma, rho = model(valid_ens1, valid_ens2, valid_amp_std)
            samples = model.sample(valid_ens1, valid_ens2, weights, sigma, rho, n_samples)
            valid_loss = energy_score(samples, valid_targets).item()

        history['train_loss'].append(np.mean(train_losses))
        history['valid_loss'].append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= hyperparams['patience']:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, history


def train_bma_all_leads(train_data: dict, valid_data: dict,
                        device: torch.device, hyperparams: dict = None,
                        n_lead_days: int = 62, n_members: int = 33) -> dict:
    """Train BMA models for all lead days."""
    models = {}
    scalers = {}

    print(f"Training Bivariate BMA for {n_lead_days} lead days ({n_members} members)...")

    for lead in range(n_lead_days):
        print(f"  Lead day {lead+1}/{n_lead_days}...", end=' ', flush=True)
        model, scaler, history = train_bma_single_lead(train_data, valid_data, lead, device, hyperparams, n_members)
        models[lead] = model
        scalers[lead] = scaler
        print(f"Valid ES: {history['valid_loss'][-1]:.4f}")

    return models, scalers


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_bma(models: dict, scalers: dict, test_data: dict, device: torch.device,
                 n_lead_days: int = 62, n_members: int = 33) -> dict:
    """Evaluate BMA models on test data."""
    n_samples_test = test_data['ensemble_rmm1'].shape[0]
    n_quantiles = len(QUANTILE_LEVELS)

    all_quantiles_rmm1 = np.zeros((n_samples_test, n_lead_days, n_quantiles))
    all_quantiles_rmm2 = np.zeros((n_samples_test, n_lead_days, n_quantiles))

    for lead in range(n_lead_days):
        model = models[lead]
        scaler = scalers[lead]
        model.eval()

        ens1 = test_data['ensemble_rmm1'][:, lead, :]
        ens2 = test_data['ensemble_rmm2'][:, lead, :]
        amp_std = test_data['amplitude_std'][:, lead]

        # Apply same normalization used during training
        test_features = np.hstack([ens1, ens2, amp_std.reshape(-1, 1)])
        test_features = scaler.transform(test_features)

        # Unstack back
        ens1 = test_features[:, :n_members]
        ens2 = test_features[:, n_members:2*n_members]
        amp_std = test_features[:, -1]

        ens1 = torch.tensor(ens1, dtype=torch.float32, device=device)
        ens2 = torch.tensor(ens2, dtype=torch.float32, device=device)
        amp_std = torch.tensor(amp_std, dtype=torch.float32, device=device)

        with torch.no_grad():
            quantiles = model.predict_quantiles(ens1, ens2, amp_std, n_samples=1000)
            quantiles = quantiles.cpu().numpy()

        all_quantiles_rmm1[:, lead, :] = quantiles[:, 0, :]
        all_quantiles_rmm2[:, lead, :] = quantiles[:, 1, :]

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
    """Run walk-forward validation for BMA."""
    if hyperparams is None:
        hyperparams = HYPERPARAMS

    # Reset seed for each run
    set_seed(SEED)

    device = get_device()
    config = DATASET_CONFIGS[dataset_name]
    test_years = config['test_years']
    n_lead_days = config['n_lead_days']
    n_members = config['n_members']
    train_window = config['train_window']

    print(f"Loading {dataset_name} ensemble data ({n_members} members)...")
    all_data = load_dataset_with_ensemble(dataset_name)

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

        models, scalers = train_bma_all_leads(train_data, valid_data, device, hyperparams, n_lead_days, n_members)

        model_path = os.path.join(save_dir, 'models', f'bma_fold_{test_year}.pt')
        torch.save({
            'models': {lead: model.state_dict() for lead, model in models.items()},
            'scalers': scalers,  # Save scalers for inference
            'hyperparams': hyperparams,
            'test_year': test_year,
            'dataset': dataset_name,
        }, model_path)
        print(f"  Models saved to {model_path}")

        results = evaluate_bma(models, scalers, test_data, device, n_lead_days, n_members)
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
        'method': 'BMA (Bivariate)',
        'dataset': dataset_name,
        'seed': SEED,
        'note': 'Uses all 33 ensemble members (more info than DBC)',
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

    results_path = os.path.join(save_dir, 'results', 'bma_summary.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {results_path}")

    print(f"\n{'='*60}")
    print(f"BMA (Bivariate) - {dataset_name} Final Summary")
    print(f"{'='*60}")
    print(f"CRPS: {summary['mean_crps']:.4f} +/- {summary['std_crps']:.4f}")
    print(f"BCOR: {summary['mean_bcor']:.4f} +/- {summary['std_bcor']:.4f}")
    print(f"RMSE: {summary['mean_rmse']:.4f} +/- {summary['std_rmse']:.4f}")
    print(f"Coverage (90%): {summary['mean_coverage_90']*100:.1f}%")
    print(f"Spread-Skill: {summary['mean_spread_skill']:.2f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Train BMA baseline')
    parser.add_argument('--dataset', type=str, default='BoM',
                        choices=['BoM', 'JMA', 'CNRM'],
                        help='Dataset to use')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results (default: auto based on dataset)')
    parser.add_argument('--lr', type=float, default=HYPERPARAMS['learning_rate'])
    parser.add_argument('--epochs', type=int, default=HYPERPARAMS['max_epochs'])
    parser.add_argument('--patience', type=int, default=HYPERPARAMS['patience'])

    args = parser.parse_args()

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
