"""
Silini's Method: Seq2Seq ANN Bias Correction
Adapted for unified data (paper_model_sqr_unified).

Simple feedforward network:
- Input: Full forecast sequence (RMM1, RMM2) flattened
- Output: Full corrected sequence
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json
import random
import argparse

DEFAULT_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Auto-detect repository root from file location
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))  # Go up 2 levels to repo root

# Dataset configurations - using unified data
DATASET_CONFIG = {
    'BoM': {
        'data_dir': os.path.join(BASE_DIR, 'data/BOM'),
        'n_lead_days': 62,
        'test_years': list(range(1999, 2014)),
        'train_window': 15,
    },
    'JMA': {
        'data_dir': os.path.join(BASE_DIR, 'data/JMA'),
        'n_lead_days': 33,
        'test_years': list(range(1998, 2013)),
        'train_window': 15,
    },
    'CNRM': {
        'data_dir': os.path.join(BASE_DIR, 'data/CNRM'),
        'n_lead_days': 60,
        'test_years': list(range(2010, 2015)),
        'train_window': 15,
    },
}

# Training config
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32
LR = 1e-3
HIDDEN_SIZE = 60
NUM_HIDDEN_LAYERS = 1


class Seq2SeqDataset(Dataset):
    """Dataset for seq2seq training."""

    def __init__(self, data_dir, years, n_lead_days):
        self.n_lead_days = n_lead_days

        self.forecast_rmm1 = np.load(os.path.join(data_dir, 'forecast_rmm1_with_day0.npy'))
        self.forecast_rmm2 = np.load(os.path.join(data_dir, 'forecast_rmm2_with_day0.npy'))
        self.gt_rmm1 = np.load(os.path.join(data_dir, 'ground_truth_rmm1.npy'))
        self.gt_rmm2 = np.load(os.path.join(data_dir, 'ground_truth_rmm2.npy'))
        self.times = np.load(os.path.join(data_dir, 'times.npy'), allow_pickle=True)

        self.datetimes = pd.to_datetime(self.times)
        self.years_array = self.datetimes.year.values

        mask = np.isin(self.years_array, years)
        self.forecast_rmm1 = self.forecast_rmm1[mask]
        self.forecast_rmm2 = self.forecast_rmm2[mask]
        self.gt_rmm1 = self.gt_rmm1[mask]
        self.gt_rmm2 = self.gt_rmm2[mask]

        self.n_forecasts = len(self.forecast_rmm1)

    def __len__(self):
        return self.n_forecasts

    def __getitem__(self, idx):
        # Input: forecast days 1 to n_lead_days
        input_rmm1 = self.forecast_rmm1[idx, 1:self.n_lead_days + 1]
        input_rmm2 = self.forecast_rmm2[idx, 1:self.n_lead_days + 1]
        # Target: ground truth for all lead days
        target_rmm1 = self.gt_rmm1[idx]
        target_rmm2 = self.gt_rmm2[idx]

        return {
            'input_rmm1': torch.FloatTensor(input_rmm1),
            'input_rmm2': torch.FloatTensor(input_rmm2),
            'target_rmm1': torch.FloatTensor(target_rmm1),
            'target_rmm2': torch.FloatTensor(target_rmm2),
            'forecast_rmm1': torch.FloatTensor(input_rmm1),
            'forecast_rmm2': torch.FloatTensor(input_rmm2),
        }


class Seq2SeqANN(nn.Module):
    """Simple feedforward network for seq2seq bias correction."""

    def __init__(self, seq_len, hidden_size=60, num_hidden_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = seq_len * 2
        self.output_size = seq_len * 2

        layers = []
        layers.append(nn.Linear(self.input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, self.output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, input_rmm1, input_rmm2):
        # Flatten: all RMM1s first, then all RMM2s
        x = torch.cat([input_rmm1, input_rmm2], dim=-1)
        output = self.network(x)
        # Split output: first half is RMM1, second half is RMM2
        pred_rmm1 = output[:, :self.seq_len]
        pred_rmm2 = output[:, self.seq_len:]
        return pred_rmm1, pred_rmm2


def bcor_fn(p1, p2, g1, g2):
    """Bivariate correlation."""
    num = (p1 * g1 + p2 * g2).sum()
    den = np.sqrt((p1**2 + p2**2).sum() * (g1**2 + g2**2).sum())
    return num / (den + 1e-8)


def train_fold(config, train_years, valid_years, test_year, device, seed):
    """Train model for one fold."""
    set_seed(seed)
    n_lead_days = config['n_lead_days']

    train_dataset = Seq2SeqDataset(config['data_dir'], train_years, n_lead_days)
    valid_dataset = Seq2SeqDataset(config['data_dir'], valid_years, n_lead_days)
    test_dataset = Seq2SeqDataset(config['data_dir'], [test_year], n_lead_days)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Seq2SeqANN(seq_len=n_lead_days, hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            input_rmm1 = batch['input_rmm1'].to(device)
            input_rmm2 = batch['input_rmm2'].to(device)
            target_rmm1 = batch['target_rmm1'].to(device)
            target_rmm2 = batch['target_rmm2'].to(device)

            optimizer.zero_grad()
            pred_rmm1, pred_rmm2 = model(input_rmm1, input_rmm2)

            # Handle NaN in targets
            valid_mask = ~(torch.isnan(target_rmm1) | torch.isnan(target_rmm2))
            if valid_mask.sum() == 0:
                continue

            pred = torch.cat([pred_rmm1, pred_rmm2], dim=-1)
            target = torch.cat([target_rmm1, target_rmm2], dim=-1)
            valid_mask_cat = torch.cat([valid_mask, valid_mask], dim=-1)
            loss = F.mse_loss(pred[valid_mask_cat], target[valid_mask_cat])
            loss.backward()
            optimizer.step()

        # Validation (using valid_loader, NOT test_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_rmm1 = batch['input_rmm1'].to(device)
                input_rmm2 = batch['input_rmm2'].to(device)
                target_rmm1 = batch['target_rmm1'].to(device)
                target_rmm2 = batch['target_rmm2'].to(device)

                pred_rmm1, pred_rmm2 = model(input_rmm1, input_rmm2)

                valid_mask = ~(torch.isnan(target_rmm1) | torch.isnan(target_rmm2))
                if valid_mask.sum() > 0:
                    pred = torch.cat([pred_rmm1, pred_rmm2], dim=-1)
                    target = torch.cat([target_rmm1, target_rmm2], dim=-1)
                    valid_mask_cat = torch.cat([valid_mask, valid_mask], dim=-1)
                    loss = F.mse_loss(pred[valid_mask_cat], target[valid_mask_cat])
                    val_loss += loss.item()

        val_loss /= max(len(valid_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_pred1, all_pred2 = [], []
    all_gt1, all_gt2 = [], []
    all_fc1, all_fc2 = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_rmm1 = batch['input_rmm1'].to(device)
            input_rmm2 = batch['input_rmm2'].to(device)

            pred_rmm1, pred_rmm2 = model(input_rmm1, input_rmm2)

            all_pred1.append(pred_rmm1.cpu().numpy())
            all_pred2.append(pred_rmm2.cpu().numpy())
            all_gt1.append(batch['target_rmm1'].numpy())
            all_gt2.append(batch['target_rmm2'].numpy())
            all_fc1.append(batch['forecast_rmm1'].numpy())
            all_fc2.append(batch['forecast_rmm2'].numpy())

    # Shape: (n_samples, n_lead_days)
    pred1 = np.concatenate(all_pred1, axis=0)
    pred2 = np.concatenate(all_pred2, axis=0)
    gt1 = np.concatenate(all_gt1, axis=0)
    gt2 = np.concatenate(all_gt2, axis=0)
    fc1 = np.concatenate(all_fc1, axis=0)
    fc2 = np.concatenate(all_fc2, axis=0)

    n_lead_days = pred1.shape[1]

    # Compute metrics PER LEAD DAY, then average across lead days
    rmse_per_lead_model = []
    rmse_per_lead_baseline = []
    bcor_per_lead_model = []
    bcor_per_lead_baseline = []

    for lead in range(n_lead_days):
        p1 = pred1[:, lead]
        p2 = pred2[:, lead]
        g1 = gt1[:, lead]
        g2 = gt2[:, lead]
        f1 = fc1[:, lead]
        f2 = fc2[:, lead]

        # Handle NaN values for this lead day
        valid = ~(np.isnan(g1) | np.isnan(g2))
        if valid.sum() == 0:
            continue

        p1, p2 = p1[valid], p2[valid]
        g1, g2 = g1[valid], g2[valid]
        f1, f2 = f1[valid], f2[valid]

        # RMSE for this lead day
        rmse_model = np.sqrt(np.mean((p1 - g1)**2 + (p2 - g2)**2))
        rmse_baseline = np.sqrt(np.mean((f1 - g1)**2 + (f2 - g2)**2))
        rmse_per_lead_model.append(rmse_model)
        rmse_per_lead_baseline.append(rmse_baseline)

        # BCOR for this lead day
        bcor_model = bcor_fn(p1, p2, g1, g2)
        bcor_baseline = bcor_fn(f1, f2, g1, g2)
        bcor_per_lead_model.append(bcor_model)
        bcor_per_lead_baseline.append(bcor_baseline)

    # Average across lead days
    avg_rmse_model = np.mean(rmse_per_lead_model)
    avg_rmse_baseline = np.mean(rmse_per_lead_baseline)
    rmse_imp = (avg_rmse_baseline - avg_rmse_model) / avg_rmse_baseline * 100

    avg_bcor_model = np.mean(bcor_per_lead_model)
    avg_bcor_baseline = np.mean(bcor_per_lead_baseline)
    bcor_imp = (avg_bcor_model - avg_bcor_baseline) / avg_bcor_baseline * 100

    return {
        'model': model,
        'rmse_imp': float(rmse_imp),
        'bcor_imp': float(bcor_imp),
        'rmse_per_lead_model': rmse_per_lead_model,
        'rmse_per_lead_baseline': rmse_per_lead_baseline,
        'bcor_per_lead_model': bcor_per_lead_model,
        'bcor_per_lead_baseline': bcor_per_lead_baseline,
        'pred_rmm1': pred1,
        'pred_rmm2': pred2,
        'gt_rmm1': gt1,
        'gt_rmm2': gt2,
        'fc_rmm1': fc1,
        'fc_rmm2': fc2,
    }


def main(dataset, seed=DEFAULT_SEED):
    set_seed(seed)
    print("=" * 70)
    print(f"SILINI'S METHOD: SEQ2SEQ ANN - {dataset} (seed={seed})")
    print("Using unified data from paper_model_sqr_unified")
    print("=" * 70)

    config = DATASET_CONFIG[dataset]
    n_lead_days = config['n_lead_days']
    test_years = config['test_years']
    train_window = config['train_window']

    print(f"Lead days: {n_lead_days}")
    print(f"Test years: {test_years[0]}-{test_years[-1]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directories include seed
    results_dir = os.path.join(BASE_DIR, 'baselines', 'silini', dataset, f'seed{seed}', 'results')
    model_dir = os.path.join(BASE_DIR, 'baselines', 'silini', dataset, f'seed{seed}', 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    all_results = []
    all_predictions = {}

    for test_year in test_years:
        print(f"\n{'=' * 60}")
        print(f"FOLD: Test Year {test_year}")

        # Proper validation split: use last year of training window as validation
        train_end = test_year - 1
        train_start = train_end - train_window + 1
        valid_year = train_end  # Last year of training window
        train_years = list(range(train_start, valid_year))  # Exclude valid_year
        valid_years = [valid_year]

        print(f"  Train: {train_start}-{valid_year-1}, Valid: {valid_year}, Test: {test_year}")

        results = train_fold(config, train_years, valid_years, test_year, device, seed)

        print(f"  RMSE Improvement: {results['rmse_imp']:.2f}%")
        print(f"  BCOR Improvement: {results['bcor_imp']:.2f}%")

        # Save model
        torch.save({
            'model_state_dict': results['model'].state_dict(),
            'test_year': test_year,
        }, os.path.join(model_dir, f'model_fold_{test_year}.pt'))

        all_results.append({
            'year': test_year,
            'rmse_imp': results['rmse_imp'],
            'bcor_imp': results['bcor_imp'],
        })

        all_predictions[f'pred_rmm1_{test_year}'] = results['pred_rmm1']
        all_predictions[f'pred_rmm2_{test_year}'] = results['pred_rmm2']
        all_predictions[f'gt_rmm1_{test_year}'] = results['gt_rmm1']
        all_predictions[f'gt_rmm2_{test_year}'] = results['gt_rmm2']
        all_predictions[f'fc_rmm1_{test_year}'] = results['fc_rmm1']
        all_predictions[f'fc_rmm2_{test_year}'] = results['fc_rmm2']

    # Save predictions
    np.savez(os.path.join(results_dir, 'predictions.npz'), **all_predictions)

    # Summary
    rmse_imps = [r['rmse_imp'] for r in all_results]
    bcor_imps = [r['bcor_imp'] for r in all_results]

    summary = {
        'model': 'silini_seq2seq_ann',
        'dataset': dataset,
        'rmse_improvement_mean': float(np.mean(rmse_imps)),
        'rmse_improvement_std': float(np.std(rmse_imps)),
        'bcor_improvement_mean': float(np.mean(bcor_imps)),
        'bcor_improvement_std': float(np.std(bcor_imps)),
        'per_year': all_results
    }

    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"SILINI - {dataset} Results:")
    print(f"  RMSE Improvement: {np.mean(rmse_imps):.2f}% +/- {np.std(rmse_imps):.2f}%")
    print(f"  BCOR Improvement: {np.mean(bcor_imps):.2f}% +/- {np.std(bcor_imps):.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['BoM', 'JMA', 'CNRM'])
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    args = parser.parse_args()
    main(args.dataset, args.seed)
