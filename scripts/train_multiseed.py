#!/usr/bin/env python3
"""
Multi-seed DBC training script for robust evaluation.

Runs the same experiment with multiple random seeds and aggregates results.
Models are saved in nested folders by seed number for easy organization.

Usage:
    # Run with default seeds (42, 123, 456, 789, 1024)
    python scripts/train_multiseed.py --dataset bom --normalize

    # Run with custom seeds
    python scripts/train_multiseed.py --dataset jma --normalize --seeds 42 123 456

    # Run single seed (useful for parallel execution)
    python scripts/train_multiseed.py --dataset cnrm --no-normalize --seeds 789

Output structure:
    BoM_norm/
        seed42/
            models/
                model_fold_1999.pt
                ...
            results/
                predictions.npz
                summary.json
        seed123/
            ...
        multiseed_summary/
            aggregated_results.json  (combined stats across all seeds)
    BoM_no_norm/
        seed42/
        ...
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (DATASETS, QUANTILE_LEVELS, N_QUANTILES, D_MODEL, NHEAD,
                    NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT, N_EPOCHS, PATIENCE,
                    LR, WEIGHT_DECAY, TRAIN_WINDOW)
from src.model import DBCTransformerNoMeta
from src.dataset import (WalkForwardDataset, ForecastBatchSampler, collate_fn,
                     compute_baseline_rmse_per_lead)
from src.loss import bivariate_dbc_loss, bivariate_dbc_loss_normalized, compute_crps_from_quantiles

# Default seeds for multi-seed experiments
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Base directory
BASE_DIR = Path(__file__).parent.parent


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_output_dir(dataset_name, normalize):
    """Get base output directory (e.g., BoM_norm/)."""
    suffix = 'norm' if normalize else 'no_norm'
    dataset_display = DATASETS[dataset_name]['name']
    return BASE_DIR / f'{dataset_display}_{suffix}'


def get_output_dir_with_seed(dataset_name, normalize, seed):
    """Get output directory for a specific seed (e.g., BoM_norm/seed42/)."""
    base_dir = get_base_output_dir(dataset_name, normalize)
    return base_dir / f'seed{seed}'


def get_multiseed_summary_dir(dataset_name, normalize):
    """Get directory for aggregated multi-seed results."""
    base_dir = get_base_output_dir(dataset_name, normalize)
    return base_dir / 'multiseed_summary'


def train_one_fold(dataset_name, test_year, normalize, device, seed):
    """Train model for one fold (test year) with specific seed."""
    set_seed(seed)  # Set seed for this fold
    config = DATASETS[dataset_name]
    data_dir = config['data_dir']
    n_lead_days = config['n_lead_days']
    max_seq_len = config['max_seq_len']
    all_years = config['all_years']

    # Walk-forward split
    valid_years = [test_year - 1]
    train_years = [y for y in all_years if y < test_year - 1]
    if len(train_years) > TRAIN_WINDOW:
        train_years = train_years[-TRAIN_WINDOW:]

    print(f"\n  Fold: Test Year {test_year}")
    print(f"    Train: {train_years[0]}-{train_years[-1]} ({len(train_years)} years)")
    print(f"    Valid: {valid_years}")

    # Create datasets
    train_dataset = WalkForwardDataset(data_dir, train_years, n_lead_days)
    valid_dataset = WalkForwardDataset(data_dir, valid_years, n_lead_days)
    test_dataset = WalkForwardDataset(data_dir, [test_year], n_lead_days)

    # Create data loaders
    train_sampler = ForecastBatchSampler(train_dataset, shuffle=True)
    valid_sampler = ForecastBatchSampler(valid_dataset, shuffle=False)
    test_sampler = ForecastBatchSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

    # Compute baseline RMSE for normalization
    baseline_rmse = compute_baseline_rmse_per_lead(train_dataset)
    baseline_rmse_tensor = torch.FloatTensor(baseline_rmse).to(device)

    # Create model
    model = DBCTransformerNoMeta(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        n_quantiles=N_QUANTILES,
        max_seq_len=max_seq_len
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    # Training loop
    best_valid_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(N_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in train_loader:
            b = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            quantile_predictions = model(
                b['input_rmm1'], b['input_rmm2'], b['input_amp'], b['attention_mask']
            )

            if normalize:
                loss = bivariate_dbc_loss_normalized(
                    quantile_predictions, b['target_rmm1'], b['target_rmm2'],
                    QUANTILE_LEVELS, b['lead_day'], baseline_rmse_tensor
                )
            else:
                loss = bivariate_dbc_loss(
                    quantile_predictions, b['target_rmm1'], b['target_rmm2'],
                    QUANTILE_LEVELS
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches
        scheduler.step()

        # Validation
        model.eval()
        valid_loss = 0
        valid_batches = 0

        with torch.no_grad():
            for batch in valid_loader:
                b = {k: v.to(device) for k, v in batch.items()}

                quantile_predictions = model(
                    b['input_rmm1'], b['input_rmm2'], b['input_amp'], b['attention_mask']
                )

                if normalize:
                    loss = bivariate_dbc_loss_normalized(
                        quantile_predictions, b['target_rmm1'], b['target_rmm2'],
                        QUANTILE_LEVELS, b['lead_day'], baseline_rmse_tensor
                    )
                else:
                    loss = bivariate_dbc_loss(
                        quantile_predictions, b['target_rmm1'], b['target_rmm2'],
                        QUANTILE_LEVELS
                    )

                valid_loss += loss.item()
                valid_batches += 1

        valid_loss /= valid_batches

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    all_quantiles_rmm1 = []
    all_quantiles_rmm2 = []
    all_gt1 = []
    all_gt2 = []
    all_fc1 = []
    all_fc2 = []
    all_lead_days = []

    with torch.no_grad():
        for batch in test_loader:
            b = {k: v.to(device) for k, v in batch.items()}

            quantile_predictions = model(
                b['input_rmm1'], b['input_rmm2'], b['input_amp'], b['attention_mask']
            )

            all_quantiles_rmm1.append(quantile_predictions[:, 0, :].cpu().numpy())
            all_quantiles_rmm2.append(quantile_predictions[:, 1, :].cpu().numpy())
            all_gt1.append(b['target_rmm1'].cpu().numpy())
            all_gt2.append(b['target_rmm2'].cpu().numpy())
            all_fc1.append(b['forecast_rmm1'].cpu().numpy())
            all_fc2.append(b['forecast_rmm2'].cpu().numpy())
            all_lead_days.append(b['lead_day'].cpu().numpy())

    # Concatenate results
    all_quantiles_rmm1 = np.concatenate(all_quantiles_rmm1)
    all_quantiles_rmm2 = np.concatenate(all_quantiles_rmm2)
    all_gt1 = np.concatenate(all_gt1)
    all_gt2 = np.concatenate(all_gt2)
    all_fc1 = np.concatenate(all_fc1)
    all_fc2 = np.concatenate(all_fc2)
    all_lead_days = np.concatenate(all_lead_days)

    # Compute metrics using median
    median_idx = QUANTILE_LEVELS.index(0.5)
    pred1 = all_quantiles_rmm1[:, median_idx]
    pred2 = all_quantiles_rmm2[:, median_idx]

    # Compute metrics PER LEAD DAY, then average
    unique_leads = np.unique(all_lead_days)
    rmse_per_lead_pred = []
    rmse_per_lead_fc = []
    bcor_per_lead_pred = []
    bcor_per_lead_fc = []

    def bcor_fn(p1, p2, g1, g2):
        num = (p1 * g1 + p2 * g2).sum()
        den = np.sqrt((p1**2 + p2**2).sum() * (g1**2 + g2**2).sum()) + 1e-8
        return num / den

    for lead in unique_leads:
        mask = all_lead_days == lead
        p1, p2 = pred1[mask], pred2[mask]
        g1, g2 = all_gt1[mask], all_gt2[mask]
        f1, f2 = all_fc1[mask], all_fc2[mask]

        valid = ~(np.isnan(g1) | np.isnan(g2))
        if valid.sum() == 0:
            continue
        p1, p2 = p1[valid], p2[valid]
        g1, g2 = g1[valid], g2[valid]
        f1, f2 = f1[valid], f2[valid]

        rmse_pred_ld = np.sqrt(np.mean((p1 - g1)**2 + (p2 - g2)**2))
        rmse_fc_ld = np.sqrt(np.mean((f1 - g1)**2 + (f2 - g2)**2))
        rmse_per_lead_pred.append(rmse_pred_ld)
        rmse_per_lead_fc.append(rmse_fc_ld)

        bcor_pred_ld = bcor_fn(p1, p2, g1, g2)
        bcor_fc_ld = bcor_fn(f1, f2, g1, g2)
        bcor_per_lead_pred.append(bcor_pred_ld)
        bcor_per_lead_fc.append(bcor_fc_ld)

    # Average across lead days
    avg_rmse_pred = np.mean(rmse_per_lead_pred)
    avg_rmse_fc = np.mean(rmse_per_lead_fc)
    rmse_improvement = (avg_rmse_fc - avg_rmse_pred) / avg_rmse_fc * 100

    avg_bcor_pred = np.mean(bcor_per_lead_pred)
    avg_bcor_fc = np.mean(bcor_per_lead_fc)
    bcor_improvement = (avg_bcor_pred - avg_bcor_fc) / avg_bcor_fc * 100

    # Coverage (90% interval)
    q05_idx = QUANTILE_LEVELS.index(0.05)
    q95_idx = QUANTILE_LEVELS.index(0.95)
    coverage1 = np.mean((all_gt1 >= all_quantiles_rmm1[:, q05_idx]) &
                        (all_gt1 <= all_quantiles_rmm1[:, q95_idx]))
    coverage2 = np.mean((all_gt2 >= all_quantiles_rmm2[:, q05_idx]) &
                        (all_gt2 <= all_quantiles_rmm2[:, q95_idx]))
    coverage = (coverage1 + coverage2) / 2

    # CRPS
    crps1 = compute_crps_from_quantiles(all_quantiles_rmm1, all_gt1, QUANTILE_LEVELS)
    crps2 = compute_crps_from_quantiles(all_quantiles_rmm2, all_gt2, QUANTILE_LEVELS)
    crps = (crps1 + crps2) / 2

    # Also compute absolute values (not just improvements)
    print(f"    BCOR: {avg_bcor_pred:.4f} (baseline: {avg_bcor_fc:.4f}, Δ={bcor_improvement:+.2f}%)")
    print(f"    RMSE: {avg_rmse_pred:.4f} (baseline: {avg_rmse_fc:.4f}, Δ={rmse_improvement:+.2f}%)")

    return {
        'test_year': test_year,
        'rmse_model': avg_rmse_pred,
        'rmse_baseline': avg_rmse_fc,
        'rmse_improvement': rmse_improvement,
        'bcor_model': avg_bcor_pred,
        'bcor_baseline': avg_bcor_fc,
        'bcor_improvement': bcor_improvement,
        'coverage': coverage,
        'crps': crps,
        'model_state': best_model_state,
        'quantiles_rmm1': all_quantiles_rmm1,
        'quantiles_rmm2': all_quantiles_rmm2,
        'gt_rmm1': all_gt1,
        'gt_rmm2': all_gt2,
        'forecast_rmm1': all_fc1,
        'forecast_rmm2': all_fc2,
        'lead_days': all_lead_days,
    }


def train_one_seed(dataset_name, normalize, device, seed, test_years=None):
    """Train all folds for one seed."""
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")

    set_seed(seed)

    config = DATASETS[dataset_name]
    output_dir = get_output_dir_with_seed(dataset_name, normalize, seed)
    models_dir = output_dir / 'models'
    results_dir = output_dir / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Get test years
    if test_years is None:
        test_years = config['test_years']

    # Train all folds
    all_results = []
    for test_year in test_years:
        result = train_one_fold(dataset_name, test_year, normalize, device, seed)
        all_results.append(result)

        # Save model
        torch.save({
            'model_state_dict': result['model_state'],
            'test_year': test_year,
            'seed': seed,
            'config': {
                'dataset': dataset_name,
                'normalize': normalize,
                'd_model': D_MODEL,
                'nhead': NHEAD,
                'num_layers': NUM_LAYERS,
                'n_quantiles': N_QUANTILES,
                'quantile_levels': QUANTILE_LEVELS,
            }
        }, models_dir / f'model_fold_{test_year}.pt')

    # Save predictions
    np.savez(
        results_dir / 'predictions.npz',
        **{f'quantiles_rmm1_{r["test_year"]}': r['quantiles_rmm1'] for r in all_results},
        **{f'quantiles_rmm2_{r["test_year"]}': r['quantiles_rmm2'] for r in all_results},
        **{f'gt_rmm1_{r["test_year"]}': r['gt_rmm1'] for r in all_results},
        **{f'gt_rmm2_{r["test_year"]}': r['gt_rmm2'] for r in all_results},
        **{f'forecast_rmm1_{r["test_year"]}': r['forecast_rmm1'] for r in all_results},
        **{f'forecast_rmm2_{r["test_year"]}': r['forecast_rmm2'] for r in all_results},
        **{f'lead_days_{r["test_year"]}': r['lead_days'] for r in all_results},
    )

    # Compute summary statistics
    rmse_improvements = [r['rmse_improvement'] for r in all_results]
    bcor_improvements = [r['bcor_improvement'] for r in all_results]
    rmse_models = [r['rmse_model'] for r in all_results]
    bcor_models = [r['bcor_model'] for r in all_results]
    rmse_baselines = [r['rmse_baseline'] for r in all_results]
    bcor_baselines = [r['bcor_baseline'] for r in all_results]
    coverages = [r['coverage'] for r in all_results]
    crps_values = [r['crps'] for r in all_results]

    summary = {
        'dataset': dataset_name,
        'normalize': normalize,
        'seed': seed,
        'n_years': len(all_results),
        # Model absolute values
        'rmse_model_mean': float(np.mean(rmse_models)),
        'rmse_model_std': float(np.std(rmse_models)),
        'bcor_model_mean': float(np.mean(bcor_models)),
        'bcor_model_std': float(np.std(bcor_models)),
        # Baseline values
        'rmse_baseline_mean': float(np.mean(rmse_baselines)),
        'bcor_baseline_mean': float(np.mean(bcor_baselines)),
        # Improvements
        'rmse_improvement_mean': float(np.mean(rmse_improvements)),
        'rmse_improvement_std': float(np.std(rmse_improvements)),
        'bcor_improvement_mean': float(np.mean(bcor_improvements)),
        'bcor_improvement_std': float(np.std(bcor_improvements)),
        # Other metrics
        'coverage_mean': float(np.mean(coverages)),
        'coverage_std': float(np.std(coverages)),
        'crps_mean': float(np.mean(crps_values)),
        'crps_std': float(np.std(crps_values)),
        'per_year': [
            {
                'year': int(r['test_year']),
                'rmse_model': float(r['rmse_model']),
                'rmse_baseline': float(r['rmse_baseline']),
                'rmse_imp': float(r['rmse_improvement']),
                'bcor_model': float(r['bcor_model']),
                'bcor_baseline': float(r['bcor_baseline']),
                'bcor_imp': float(r['bcor_improvement']),
                'coverage': float(r['coverage']),
                'crps': float(r['crps']),
            }
            for r in all_results
        ]
    }

    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Seed {seed} Summary:")
    print(f"    BCOR: {np.mean(bcor_models):.4f} (Δ={np.mean(bcor_improvements):+.2f}% ± {np.std(bcor_improvements):.2f}%)")
    print(f"    RMSE: {np.mean(rmse_models):.4f} (Δ={np.mean(rmse_improvements):+.2f}% ± {np.std(rmse_improvements):.2f}%)")
    print(f"    Coverage: {np.mean(coverages)*100:.2f}%, CRPS: {np.mean(crps_values):.4f}")

    return summary


def aggregate_multiseed_results(dataset_name, normalize, seeds, summaries):
    """Aggregate results across all seeds and save summary."""
    summary_dir = get_multiseed_summary_dir(dataset_name, normalize)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics from each seed
    bcor_improvements = [s['bcor_improvement_mean'] for s in summaries]
    rmse_improvements = [s['rmse_improvement_mean'] for s in summaries]
    bcor_models = [s['bcor_model_mean'] for s in summaries]
    rmse_models = [s['rmse_model_mean'] for s in summaries]
    coverages = [s['coverage_mean'] for s in summaries]
    crps_values = [s['crps_mean'] for s in summaries]

    aggregated = {
        'dataset': dataset_name,
        'normalize': normalize,
        'seeds': seeds,
        'n_seeds': len(seeds),
        # Aggregated across seeds (mean of per-seed means)
        'bcor_model': {
            'mean': float(np.mean(bcor_models)),
            'std': float(np.std(bcor_models)),
            'min': float(np.min(bcor_models)),
            'max': float(np.max(bcor_models)),
        },
        'rmse_model': {
            'mean': float(np.mean(rmse_models)),
            'std': float(np.std(rmse_models)),
            'min': float(np.min(rmse_models)),
            'max': float(np.max(rmse_models)),
        },
        'bcor_improvement': {
            'mean': float(np.mean(bcor_improvements)),
            'std': float(np.std(bcor_improvements)),
            'min': float(np.min(bcor_improvements)),
            'max': float(np.max(bcor_improvements)),
        },
        'rmse_improvement': {
            'mean': float(np.mean(rmse_improvements)),
            'std': float(np.std(rmse_improvements)),
            'min': float(np.min(rmse_improvements)),
            'max': float(np.max(rmse_improvements)),
        },
        'coverage': {
            'mean': float(np.mean(coverages)),
            'std': float(np.std(coverages)),
        },
        'crps': {
            'mean': float(np.mean(crps_values)),
            'std': float(np.std(crps_values)),
        },
        # Per-seed summaries
        'per_seed': {
            seed: {
                'bcor_model': s['bcor_model_mean'],
                'bcor_improvement': s['bcor_improvement_mean'],
                'rmse_model': s['rmse_model_mean'],
                'rmse_improvement': s['rmse_improvement_mean'],
                'coverage': s['coverage_mean'],
                'crps': s['crps_mean'],
            }
            for seed, s in zip(seeds, summaries)
        }
    }

    with open(summary_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print final summary
    config = DATASETS[dataset_name]
    print(f"\n{'='*70}")
    print(f"MULTI-SEED SUMMARY - {config['name']} ({'Normalized' if normalize else 'Standard'})")
    print(f"{'='*70}")
    print(f"Seeds: {seeds}")
    print(f"\nBCOR Model:       {np.mean(bcor_models):.4f} ± {np.std(bcor_models):.4f} (range: {np.min(bcor_models):.4f} - {np.max(bcor_models):.4f})")
    print(f"BCOR Improvement: {np.mean(bcor_improvements):+.2f}% ± {np.std(bcor_improvements):.2f}%")
    print(f"RMSE Model:       {np.mean(rmse_models):.4f} ± {np.std(rmse_models):.4f}")
    print(f"RMSE Improvement: {np.mean(rmse_improvements):+.2f}% ± {np.std(rmse_improvements):.2f}%")
    print(f"Coverage:         {np.mean(coverages)*100:.2f}% ± {np.std(coverages)*100:.2f}%")
    print(f"CRPS:             {np.mean(crps_values):.4f} ± {np.std(crps_values):.4f}")
    print(f"\nResults saved to: {summary_dir}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Multi-seed DBC Training for Robust Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all 5 default seeds on BoM with normalization
    python scripts/train_multiseed.py --dataset bom --normalize

    # Run specific seeds on JMA without normalization
    python scripts/train_multiseed.py --dataset jma --no-normalize --seeds 42 123 456

    # Run a single seed (for parallel execution on cluster)
    python scripts/train_multiseed.py --dataset cnrm --normalize --seeds 789 --single-seed
        """
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['bom', 'jma', 'cnrm'],
                        help='Dataset to train on')
    parser.add_argument('--normalize', action='store_true',
                        help='Use normalized loss')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Use standard loss (no normalization)')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'Random seeds to use (default: {DEFAULT_SEEDS})')
    parser.add_argument('--single-seed', action='store_true',
                        help='Run only the first seed (for parallel execution)')
    parser.add_argument('--test_year', type=int, default=None,
                        help='Single test year (default: all test years)')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()

    # Handle single seed mode
    if args.single_seed:
        seeds = [args.seeds[0]]
    else:
        seeds = args.seeds

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = DATASETS[args.dataset]
    test_years = [args.test_year] if args.test_year else None

    print(f"\n{'#'*70}")
    print(f"# MULTI-SEED DBC TRAINING")
    print(f"# Dataset: {config['name']}")
    print(f"# Normalized: {args.normalize}")
    print(f"# Seeds: {seeds}")
    print(f"{'#'*70}")

    # Train for each seed
    summaries = []
    for seed in seeds:
        summary = train_one_seed(args.dataset, args.normalize, device, seed, test_years)
        summaries.append(summary)

    # Aggregate results (only if running multiple seeds)
    if len(seeds) > 1:
        aggregate_multiseed_results(args.dataset, args.normalize, seeds, summaries)


if __name__ == '__main__':
    main()
