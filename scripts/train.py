#!/usr/bin/env python3
"""
Unified DBC training script for BoM, JMA, and CNRM datasets.

Usage:
    python scripts/train.py --dataset bom --normalize
    python scripts/train.py --dataset jma --no-normalize
    python scripts/train.py --dataset cnrm --normalize --test_year 2012
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (DATASETS, QUANTILE_LEVELS, N_QUANTILES, D_MODEL, NHEAD,
                    NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT, N_EPOCHS, PATIENCE,
                    LR, WEIGHT_DECAY, TRAIN_WINDOW, get_output_dir, SEED, set_seed)

# Set seed for reproducibility
set_seed(SEED)
from src.model import DBCTransformerNoMeta
from src.dataset import (WalkForwardDataset, ForecastBatchSampler, collate_fn,
                     compute_baseline_rmse_per_lead)
from src.loss import bivariate_dbc_loss, bivariate_dbc_loss_normalized, compute_crps_from_quantiles


def train_one_fold(dataset_name, test_year, normalize, device):
    """Train model for one fold (test year)."""
    set_seed(SEED)  # Reset seed for each fold
    config = DATASETS[dataset_name]
    data_dir = config['data_dir']
    n_lead_days = config['n_lead_days']
    max_seq_len = config['max_seq_len']
    all_years = config['all_years']

    # Walk-forward split (1-year validation, same as baselines)
    valid_years = [test_year - 1]
    train_years = [y for y in all_years if y < test_year - 1]
    if len(train_years) > TRAIN_WINDOW:
        train_years = train_years[-TRAIN_WINDOW:]

    print(f"\n{'='*60}")
    print(f"Fold: Test Year {test_year}")
    print(f"{'='*60}")
    print(f"  Train: {train_years[0]}-{train_years[-1]} ({len(train_years)} years)")
    print(f"  Valid: {valid_years}")
    print(f"  Test: {test_year}")

    # Create datasets
    train_dataset = WalkForwardDataset(data_dir, train_years, n_lead_days)
    valid_dataset = WalkForwardDataset(data_dir, valid_years, n_lead_days)
    test_dataset = WalkForwardDataset(data_dir, [test_year], n_lead_days)

    # Create data loaders (using dataset to handle NaN-filtered samples)
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
            # Move to device
            b = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass
            quantile_predictions = model(
                b['input_rmm1'], b['input_rmm2'], b['input_amp'], b['attention_mask']
            )

            # Compute loss
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

            # Backward pass
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

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train={train_loss:.4f}, Valid={valid_loss:.4f}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
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

    # Compute metrics PER LEAD DAY, then average across lead days
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

        # Handle NaN for this lead day
        valid = ~(np.isnan(g1) | np.isnan(g2))
        if valid.sum() == 0:
            continue
        p1, p2 = p1[valid], p2[valid]
        g1, g2 = g1[valid], g2[valid]
        f1, f2 = f1[valid], f2[valid]

        # RMSE for this lead day
        rmse_pred_ld = np.sqrt(np.mean((p1 - g1)**2 + (p2 - g2)**2))
        rmse_fc_ld = np.sqrt(np.mean((f1 - g1)**2 + (f2 - g2)**2))
        rmse_per_lead_pred.append(rmse_pred_ld)
        rmse_per_lead_fc.append(rmse_fc_ld)

        # BCOR for this lead day
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

    print(f"  RMSE Imp: {rmse_improvement:.2f}%, BCOR Imp: {bcor_improvement:.2f}%")
    print(f"  Coverage: {coverage*100:.2f}%, CRPS: {crps:.4f}")

    return {
        'test_year': test_year,
        'rmse_improvement': rmse_improvement,
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


def main():
    parser = argparse.ArgumentParser(description='Unified DBC Training')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['bom', 'jma', 'cnrm'],
                        help='Dataset to train on')
    parser.add_argument('--normalize', action='store_true',
                        help='Use normalized loss')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Use standard loss (no normalization)')
    parser.add_argument('--test_year', type=int, default=None,
                        help='Single test year (default: all test years)')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = DATASETS[args.dataset]
    output_dir = get_output_dir(args.dataset, args.normalize)
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DBC Training: {config['name']}")
    print(f"Normalized: {args.normalize}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Get test years
    if args.test_year:
        test_years = [args.test_year]
    else:
        test_years = config['test_years']

    # Train all folds
    all_results = []
    for test_year in test_years:
        result = train_one_fold(args.dataset, test_year, args.normalize, device)
        all_results.append(result)

        # Save model
        torch.save({
            'model_state_dict': result['model_state'],
            'test_year': test_year,
            'config': {
                'dataset': args.dataset,
                'normalize': args.normalize,
                'd_model': D_MODEL,
                'nhead': NHEAD,
                'num_layers': NUM_LAYERS,
                'n_quantiles': N_QUANTILES,
                'quantile_levels': QUANTILE_LEVELS,
            }
        }, os.path.join(models_dir, f'model_fold_{test_year}.pt'))

    # Save predictions
    np.savez(
        os.path.join(results_dir, 'predictions.npz'),
        **{f'quantiles_rmm1_{r["test_year"]}': r['quantiles_rmm1'] for r in all_results},
        **{f'quantiles_rmm2_{r["test_year"]}': r['quantiles_rmm2'] for r in all_results},
        **{f'gt_rmm1_{r["test_year"]}': r['gt_rmm1'] for r in all_results},
        **{f'gt_rmm2_{r["test_year"]}': r['gt_rmm2'] for r in all_results},
        **{f'forecast_rmm1_{r["test_year"]}': r['forecast_rmm1'] for r in all_results},
        **{f'forecast_rmm2_{r["test_year"]}': r['forecast_rmm2'] for r in all_results},
        **{f'lead_days_{r["test_year"]}': r['lead_days'] for r in all_results},
    )

    # Summary
    rmse_improvements = [r['rmse_improvement'] for r in all_results]
    bcor_improvements = [r['bcor_improvement'] for r in all_results]
    coverages = [r['coverage'] for r in all_results]
    crps_values = [r['crps'] for r in all_results]

    summary = {
        'dataset': args.dataset,
        'normalize': args.normalize,
        'rmse_improvement_mean': float(np.mean(rmse_improvements)),
        'rmse_improvement_std': float(np.std(rmse_improvements)),
        'bcor_improvement_mean': float(np.mean(bcor_improvements)),
        'bcor_improvement_std': float(np.std(bcor_improvements)),
        'coverage_mean': float(np.mean(coverages)),
        'coverage_std': float(np.std(coverages)),
        'crps_mean': float(np.mean(crps_values)),
        'crps_std': float(np.std(crps_values)),
        'per_year': [
            {
                'year': int(r['test_year']),
                'rmse_imp': float(r['rmse_improvement']),
                'bcor_imp': float(r['bcor_improvement']),
                'coverage': float(r['coverage']),
                'crps': float(r['crps']),
            }
            for r in all_results
        ]
    }

    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY - {config['name']} ({'Normalized' if args.normalize else 'Standard'} Loss)")
    print(f"{'='*60}")
    print(f"RMSE Improvement: {np.mean(rmse_improvements):.2f}% +/- {np.std(rmse_improvements):.2f}%")
    print(f"BCOR Improvement: {np.mean(bcor_improvements):.2f}% +/- {np.std(bcor_improvements):.2f}%")
    print(f"Coverage (90%): {np.mean(coverages)*100:.2f}% +/- {np.std(coverages)*100:.2f}%")
    print(f"CRPS: {np.mean(crps_values):.4f} +/- {np.std(crps_values):.4f}")
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()
