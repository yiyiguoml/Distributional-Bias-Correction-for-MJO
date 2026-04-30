"""
Kim's Method: Point-to-Point LSTM Bias Correction
Adapted for unified data (paper_model_sqr_unified).

Trains separate LSTM models per fold:
- N_lead_days x 4 phase categories = N models per fold
- Each model takes RMM1/RMM2 forecast at that lead day for that phase
- Predicts corrected RMM1/RMM2 values (direct learning)
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
BATCH_SIZE = 64
LR = 1e-3
HIDDEN_SIZE = 32
NUM_LAYERS = 1
N_PHASE_CATEGORIES = 4


def compute_phase(rmm1, rmm2):
    """Compute MJO phase (1-8) from RMM1, RMM2 values."""
    angle = np.arctan2(rmm2, rmm1)
    # Convert angle (-pi to pi) to phase (1-8), each phase = 45 degrees
    phase = np.floor((angle + np.pi) / (np.pi / 4)) + 1
    phase = np.clip(phase, 1, 8).astype(int)
    return phase


def get_phase_category(phase):
    """Map MJO phase (1-8) to category (0-3). MJO phases are circular."""
    if phase in [8, 1]:
        return 0
    elif phase in [2, 3]:
        return 1
    elif phase in [4, 5]:
        return 2
    else:  # 6, 7
        return 3


class PointDataset(Dataset):
    """Dataset for point-to-point training at a specific lead day and phase category."""

    def __init__(self, data_dir, years, lead_day, phase_category, n_lead_days):
        self.lead_day = lead_day
        self.phase_category = phase_category
        self.n_lead_days = n_lead_days

        self.forecast_rmm1 = np.load(os.path.join(data_dir, 'forecast_rmm1_with_day0.npy'))
        self.forecast_rmm2 = np.load(os.path.join(data_dir, 'forecast_rmm2_with_day0.npy'))
        self.gt_rmm1 = np.load(os.path.join(data_dir, 'ground_truth_rmm1.npy'))
        self.gt_rmm2 = np.load(os.path.join(data_dir, 'ground_truth_rmm2.npy'))
        self.times = np.load(os.path.join(data_dir, 'times.npy'), allow_pickle=True)

        # Compute phase from FORECAST at the specific lead day (not day 0!)
        forecast_rmm1_at_lead = self.forecast_rmm1[:, lead_day]
        forecast_rmm2_at_lead = self.forecast_rmm2[:, lead_day]
        self.phase_at_lead = compute_phase(forecast_rmm1_at_lead, forecast_rmm2_at_lead)
        self.phase_categories = np.array([get_phase_category(p) for p in self.phase_at_lead])

        self.datetimes = pd.to_datetime(self.times)
        self.years_array = self.datetimes.year.values

        year_mask = np.isin(self.years_array, years)
        phase_mask = self.phase_categories == phase_category

        # Also filter out NaN targets
        nan_mask = ~(np.isnan(self.gt_rmm1[:, lead_day - 1]) | np.isnan(self.gt_rmm2[:, lead_day - 1]))
        mask = year_mask & phase_mask & nan_mask

        self.forecast_rmm1 = self.forecast_rmm1[mask]
        self.forecast_rmm2 = self.forecast_rmm2[mask]
        self.gt_rmm1 = self.gt_rmm1[mask]
        self.gt_rmm2 = self.gt_rmm2[mask]

        self.n_samples = len(self.forecast_rmm1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        input_rmm1 = self.forecast_rmm1[idx, self.lead_day]
        input_rmm2 = self.forecast_rmm2[idx, self.lead_day]
        target_rmm1 = self.gt_rmm1[idx, self.lead_day - 1]
        target_rmm2 = self.gt_rmm2[idx, self.lead_day - 1]

        return {
            'input': torch.FloatTensor([input_rmm1, input_rmm2]),
            'target': torch.FloatTensor([target_rmm1, target_rmm2]),
            'forecast': torch.FloatTensor([input_rmm1, input_rmm2]),
        }


class PointLSTM(nn.Module):
    """Simple LSTM for point-to-point bias correction."""

    def __init__(self, input_size=2, hidden_size=32, num_layers=1, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x_input = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x_input)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


def train_model_for_lead_phase(config, lead_day, phase_category, train_years, valid_years, device):
    """Train a single model for one lead day and phase category."""
    train_dataset = PointDataset(config['data_dir'], train_years, lead_day, phase_category, config['n_lead_days'])
    valid_dataset = PointDataset(config['data_dir'], valid_years, lead_day, phase_category, config['n_lead_days'])

    # Only skip if literally no samples (can't train with 0 data)
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        return None, float('inf')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PointLSTM(input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_valid_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_valid_loss


def bcor_fn(p1, p2, g1, g2):
    """Bivariate correlation."""
    num = (p1 * g1 + p2 * g2).sum()
    den = np.sqrt((p1**2 + p2**2).sum() * (g1**2 + g2**2).sum())
    return num / (den + 1e-8)


def evaluate_models(models, config, test_years, device):
    """Evaluate all models on test set."""
    data_dir = config['data_dir']
    n_lead_days = config['n_lead_days']

    forecast_rmm1 = np.load(os.path.join(data_dir, 'forecast_rmm1_with_day0.npy'))
    forecast_rmm2 = np.load(os.path.join(data_dir, 'forecast_rmm2_with_day0.npy'))
    gt_rmm1 = np.load(os.path.join(data_dir, 'ground_truth_rmm1.npy'))
    gt_rmm2 = np.load(os.path.join(data_dir, 'ground_truth_rmm2.npy'))
    times = np.load(os.path.join(data_dir, 'times.npy'), allow_pickle=True)

    datetimes = pd.to_datetime(times)
    years_array = datetimes.year.values
    mask = np.isin(years_array, test_years)

    forecast_rmm1 = forecast_rmm1[mask]
    forecast_rmm2 = forecast_rmm2[mask]
    gt_rmm1 = gt_rmm1[mask]
    gt_rmm2 = gt_rmm2[mask]

    n_samples = len(forecast_rmm1)
    # Initialize with NaN - samples with missing models will be skipped
    pred_rmm1 = np.full((n_samples, n_lead_days), np.nan)
    pred_rmm2 = np.full((n_samples, n_lead_days), np.nan)

    for lead_day in range(1, n_lead_days + 1):
        # Compute phase from FORECAST at this specific lead day
        phase_at_lead = compute_phase(forecast_rmm1[:, lead_day], forecast_rmm2[:, lead_day])
        phase_categories = np.array([get_phase_category(p) for p in phase_at_lead])

        for phase_cat in range(N_PHASE_CATEGORIES):
            model = models[lead_day - 1][phase_cat]
            phase_mask = phase_categories == phase_cat

            # Skip samples with missing models (leave as NaN to exclude from metrics)
            if model is None or phase_mask.sum() == 0:
                continue

            model.eval()
            inputs = torch.FloatTensor(np.stack([
                forecast_rmm1[phase_mask, lead_day],
                forecast_rmm2[phase_mask, lead_day]
            ], axis=1)).to(device)

            with torch.no_grad():
                outputs = model(inputs).cpu().numpy()

            pred_rmm1[phase_mask, lead_day - 1] = outputs[:, 0]
            pred_rmm2[phase_mask, lead_day - 1] = outputs[:, 1]

    # Compute metrics PER LEAD DAY, then average across lead days
    rmse_per_lead_model = []
    rmse_per_lead_baseline = []
    bcor_per_lead_model = []
    bcor_per_lead_baseline = []

    fc_rmm1 = forecast_rmm1[:, 1:n_lead_days+1]
    fc_rmm2 = forecast_rmm2[:, 1:n_lead_days+1]

    for lead in range(n_lead_days):
        p1 = pred_rmm1[:, lead]
        p2 = pred_rmm2[:, lead]
        g1 = gt_rmm1[:, lead]
        g2 = gt_rmm2[:, lead]
        f1 = fc_rmm1[:, lead]
        f2 = fc_rmm2[:, lead]

        # Handle NaN values for this lead day (in both predictions and ground truth)
        valid = ~(np.isnan(g1) | np.isnan(g2) | np.isnan(p1) | np.isnan(p2))
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
        'rmse_imp': rmse_imp,
        'bcor_imp': bcor_imp,
        'rmse_per_lead_model': rmse_per_lead_model,
        'rmse_per_lead_baseline': rmse_per_lead_baseline,
        'bcor_per_lead_model': bcor_per_lead_model,
        'bcor_per_lead_baseline': bcor_per_lead_baseline,
        'pred_rmm1': pred_rmm1,
        'pred_rmm2': pred_rmm2,
        'gt_rmm1': gt_rmm1,
        'gt_rmm2': gt_rmm2,
        'fc_rmm1': fc_rmm1,
        'fc_rmm2': fc_rmm2,
    }


def main(dataset, seed=DEFAULT_SEED):
    set_seed(seed)
    print("=" * 70)
    print(f"KIM'S METHOD: POINT-TO-POINT LSTM - {dataset} (seed={seed})")
    print("Using unified data from paper_model_sqr_unified")
    print("=" * 70)

    config = DATASET_CONFIG[dataset]
    n_lead_days = config['n_lead_days']
    test_years = config['test_years']
    train_window = config['train_window']

    print(f"Lead days: {n_lead_days}")
    print(f"Test years: {test_years[0]}-{test_years[-1]}")
    print(f"Models per fold: {n_lead_days} x 4 = {n_lead_days * 4}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directories include seed
    results_dir = os.path.join(BASE_DIR, 'baselines', 'kim', dataset, f'seed{seed}', 'results')
    model_dir = os.path.join(BASE_DIR, 'baselines', 'kim', dataset, f'seed{seed}', 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    all_results = []
    all_predictions = {}

    for test_year in test_years:
        set_seed(seed)
        print(f"\n{'=' * 60}")
        print(f"FOLD: Test Year {test_year}")

        # Proper validation split: use last year of training window as validation
        train_end = test_year - 1
        train_start = train_end - train_window + 1
        valid_year = train_end  # Last year of training window
        train_years = list(range(train_start, valid_year))  # Exclude valid_year
        valid_years = [valid_year]

        print(f"  Train: {train_start}-{valid_year-1}, Valid: {valid_year}, Test: {test_year}")

        # Train models
        models = [[None for _ in range(N_PHASE_CATEGORIES)] for _ in range(n_lead_days)]

        for lead_day in range(1, n_lead_days + 1):
            for phase_cat in range(N_PHASE_CATEGORIES):
                model, _ = train_model_for_lead_phase(config, lead_day, phase_cat, train_years, valid_years, device)
                models[lead_day - 1][phase_cat] = model

            if lead_day % 10 == 0:
                print(f"    Trained lead day {lead_day}/{n_lead_days}")

        # Save models for this fold
        fold_models = {}
        for lead_day in range(1, n_lead_days + 1):
            for phase_cat in range(N_PHASE_CATEGORIES):
                model = models[lead_day - 1][phase_cat]
                if model is not None:
                    fold_models[f'lead{lead_day}_phase{phase_cat}'] = model.state_dict()
        torch.save({
            'models': fold_models,
            'test_year': test_year,
        }, os.path.join(model_dir, f'model_fold_{test_year}.pt'))

        # Evaluate
        results = evaluate_models(models, config, [test_year], device)

        print(f"  RMSE Improvement: {results['rmse_imp']:.2f}%")
        print(f"  BCOR Improvement: {results['bcor_imp']:.2f}%")

        all_results.append({
            'year': test_year,
            'rmse_imp': float(results['rmse_imp']),
            'bcor_imp': float(results['bcor_imp']),
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
        'model': 'kim_point2point_lstm',
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
    print(f"KIM - {dataset} Results:")
    print(f"  RMSE Improvement: {np.mean(rmse_imps):.2f}% +/- {np.std(rmse_imps):.2f}%")
    print(f"  BCOR Improvement: {np.mean(bcor_imps):.2f}% +/- {np.std(bcor_imps):.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['BoM', 'JMA', 'CNRM'])
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    args = parser.parse_args()
    main(args.dataset, args.seed)
