"""
UAR Method: Transformer with Last Hidden Extraction and MSE + Cosine Similarity Loss
Adapted for unified data (paper_model_sqr_unified).

Architecture: Seq2PointTransformer with last hidden extraction.
Loss function: MSE + lambda * (1 - cosine_similarity)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from collections import defaultdict
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

# Model configuration (Original UAR: Transformer + Last Hidden)
D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 1
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# Loss parameters
P = 2
LAMBDA_VALUES = [0.0, 0.1, 0.2]  # Lambda selection

# Training config
N_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32
LR = 1e-3
VAL_RATIO = 0.15  # 15% of samples for stratified validation


class Seq2PointDataset(Dataset):
    """Dataset for sequence-to-point training."""

    def __init__(self, data_dir, years, n_lead_days):
        self.n_lead_days = n_lead_days

        self.forecast_rmm1 = np.load(os.path.join(data_dir, 'forecast_rmm1_with_day0.npy'))
        self.forecast_rmm2 = np.load(os.path.join(data_dir, 'forecast_rmm2_with_day0.npy'))
        self.gt_rmm1 = np.load(os.path.join(data_dir, 'ground_truth_rmm1.npy'))
        self.gt_rmm2 = np.load(os.path.join(data_dir, 'ground_truth_rmm2.npy'))
        self.amplitude_forecast = np.load(os.path.join(data_dir, 'amplitude_forecast.npy'))
        self.day0_info = np.load(os.path.join(data_dir, 'day0_info.npy'))
        self.times = np.load(os.path.join(data_dir, 'times.npy'), allow_pickle=True)

        self.datetimes = pd.to_datetime(self.times)
        self.years_array = self.datetimes.year.values

        mask = np.isin(self.years_array, years)
        self.forecast_rmm1 = self.forecast_rmm1[mask]
        self.forecast_rmm2 = self.forecast_rmm2[mask]
        self.gt_rmm1 = self.gt_rmm1[mask]
        self.gt_rmm2 = self.gt_rmm2[mask]
        self.amplitude_forecast = self.amplitude_forecast[mask]
        self.day0_info = self.day0_info[mask]

        # Day 0 amplitude from column 3 (actual amplitude)
        self.day0_amplitude = self.day0_info[:, 3]

        # Build full amplitude sequence
        self.forecast_amplitude_full = np.zeros((len(self.forecast_rmm1), self.n_lead_days + 1))
        self.forecast_amplitude_full[:, 0] = self.day0_amplitude
        self.forecast_amplitude_full[:, 1:] = self.amplitude_forecast[:, :self.n_lead_days]

        self.n_forecasts = len(self.forecast_rmm1)

        # Store filtered years for stratification
        self.filtered_years = self.years_array[mask]

        # Filter out samples with NaN targets
        self.samples = []
        self.sample_years = []  # Track year for each sample
        for fc in range(self.n_forecasts):
            for ld in range(1, self.n_lead_days + 1):
                gt1 = self.gt_rmm1[fc, ld - 1]
                gt2 = self.gt_rmm2[fc, ld - 1]
                if not (np.isnan(gt1) or np.isnan(gt2)):
                    self.samples.append((fc, ld))
                    self.sample_years.append(self.filtered_years[fc])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fc_idx, lead_day = self.samples[idx]
        seq_len = lead_day + 1

        input_rmm1 = self.forecast_rmm1[fc_idx, :seq_len]
        input_rmm2 = self.forecast_rmm2[fc_idx, :seq_len]
        input_amp = self.forecast_amplitude_full[fc_idx, :seq_len]
        input_day_idx = np.arange(seq_len)

        target_rmm1 = self.gt_rmm1[fc_idx, lead_day - 1]
        target_rmm2 = self.gt_rmm2[fc_idx, lead_day - 1]
        forecast_rmm1 = self.forecast_rmm1[fc_idx, lead_day]
        forecast_rmm2 = self.forecast_rmm2[fc_idx, lead_day]

        return {
            'input_rmm1': torch.FloatTensor(input_rmm1),
            'input_rmm2': torch.FloatTensor(input_rmm2),
            'input_amp': torch.FloatTensor(input_amp),
            'input_day_idx': torch.LongTensor(input_day_idx),
            'target_rmm1': torch.FloatTensor([target_rmm1]),
            'target_rmm2': torch.FloatTensor([target_rmm2]),
            'forecast_rmm1': torch.FloatTensor([forecast_rmm1]),
            'forecast_rmm2': torch.FloatTensor([forecast_rmm2]),
            'lead_day': torch.LongTensor([lead_day]),
        }

    def get_stratified_split(self, val_ratio, seed):
        """Return train and val indices, stratified by year."""
        year_to_indices = defaultdict(list)
        for idx in range(len(self.samples)):
            year = self.sample_years[idx]
            year_to_indices[year].append(idx)

        train_indices, val_indices = [], []
        rng = np.random.RandomState(seed)
        for year in sorted(year_to_indices.keys()):
            indices = year_to_indices[year].copy()
            rng.shuffle(indices)
            n_val = max(1, int(len(indices) * val_ratio))  # At least 1 sample for val
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        return train_indices, val_indices


class ForecastBatchSampler(Sampler):
    """Batches samples by forecast."""

    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group sample indices by forecast
        self.forecast_to_indices = {}
        for idx, (fc_idx, lead_day) in enumerate(dataset.samples):
            if fc_idx not in self.forecast_to_indices:
                self.forecast_to_indices[fc_idx] = []
            self.forecast_to_indices[fc_idx].append(idx)

        self.forecast_order = list(self.forecast_to_indices.keys())

    def __iter__(self):
        order = self.forecast_order.copy()
        if self.shuffle:
            np.random.shuffle(order)

        # Batch multiple forecasts together
        for i in range(0, len(order), self.batch_size):
            batch_forecasts = order[i:i + self.batch_size]
            batch_indices = []
            for fc_idx in batch_forecasts:
                batch_indices.extend(self.forecast_to_indices[fc_idx])
            yield batch_indices

    def __len__(self):
        return (len(self.forecast_order) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    """Collate with padding."""
    input_rmm1 = nn.utils.rnn.pad_sequence([b['input_rmm1'] for b in batch], batch_first=True)
    input_rmm2 = nn.utils.rnn.pad_sequence([b['input_rmm2'] for b in batch], batch_first=True)
    input_amp = nn.utils.rnn.pad_sequence([b['input_amp'] for b in batch], batch_first=True)
    input_day_idx = nn.utils.rnn.pad_sequence([b['input_day_idx'] for b in batch], batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(
        [torch.ones(len(b['input_rmm1'])) for b in batch], batch_first=True, padding_value=0
    )

    return {
        'input_rmm1': input_rmm1,
        'input_rmm2': input_rmm2,
        'input_amp': input_amp,
        'input_day_idx': input_day_idx,
        'attention_mask': attention_mask,
        'target_rmm1': torch.cat([b['target_rmm1'] for b in batch]),
        'target_rmm2': torch.cat([b['target_rmm2'] for b in batch]),
        'forecast_rmm1': torch.cat([b['forecast_rmm1'] for b in batch]),
        'forecast_rmm2': torch.cat([b['forecast_rmm2'] for b in batch]),
        'lead_day': torch.cat([b['lead_day'] for b in batch]),
    }


class Seq2PointTransformer(nn.Module):
    """Transformer for sequence-to-point bias correction with last hidden extraction."""

    def __init__(self, d_model=64, nhead=2, num_layers=1, dim_feedforward=256, dropout=0.1, max_len=64):
        super().__init__()
        self.d_model = d_model
        # 4 input features: RMM1, RMM2, Amplitude, Day_Index
        self.input_proj = nn.Linear(4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_rmm1, input_rmm2, input_amp, input_day_idx, attention_mask):
        batch_size, seq_len = input_rmm1.shape

        # Stack 4 features: RMM1, RMM2, Amplitude, Day_Index
        day_idx_float = input_day_idx.float()
        x = torch.stack([input_rmm1, input_rmm2, input_amp, day_idx_float], dim=-1)
        x = self.input_proj(x)
        x = self.dropout(x)

        src_key_padding_mask = (attention_mask == 0)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Extract last hidden state (at the target lead day position)
        seq_lengths = attention_mask.sum(dim=1).long()
        last_hidden = x[torch.arange(batch_size, device=x.device), seq_lengths - 1]
        output = self.output_head(last_hidden)

        return output


def uar_loss(pred_rmm1, pred_rmm2, target_rmm1, target_rmm2, p=2, lam=0.1):
    """UAR loss: MSE + lambda * (1 - cosine_similarity)"""
    valid = ~(torch.isnan(target_rmm1) | torch.isnan(target_rmm2))
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_rmm1.device), torch.tensor(0.0), torch.tensor(0.0)

    pred1, pred2 = pred_rmm1[valid], pred_rmm2[valid]
    tgt1, tgt2 = target_rmm1[valid], target_rmm2[valid]

    # MSE component
    A1 = torch.abs(pred1 - tgt1) ** p
    A2 = torch.abs(pred2 - tgt2) ** p
    error_mse = torch.mean(A1 + A2)

    # Cosine similarity component
    norm_pred = torch.sqrt(pred1**2 + pred2**2 + 1e-8)
    norm_obs = torch.sqrt(tgt1**2 + tgt2**2 + 1e-8)
    dot_product = pred1 * tgt1 + pred2 * tgt2
    cosine_sim = dot_product / (norm_pred * norm_obs)
    error_cos = torch.mean(1 - cosine_sim)

    loss = error_mse + lam * error_cos
    return loss, error_mse, error_cos


def bcor_fn(p1, p2, g1, g2):
    """Bivariate correlation."""
    num = (p1 * g1 + p2 * g2).sum()
    den = np.sqrt((p1**2 + p2**2).sum() * (g1**2 + g2**2).sum())
    return num / (den + 1e-8)


def train_single_model(config, train_years, lam, device, seed):
    """Train a single model with a specific lambda value using stratified validation."""
    set_seed(seed)
    n_lead_days = config['n_lead_days']

    # Create dataset with all training years
    full_dataset = Seq2PointDataset(config['data_dir'], train_years, n_lead_days)

    # Stratified split by year
    train_indices, val_indices = full_dataset.get_stratified_split(VAL_RATIO, seed)

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, val_indices)

    # Use simple DataLoader with shuffling for subsets
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE * n_lead_days, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * n_lead_days, shuffle=False, collate_fn=collate_fn)

    model = Seq2PointTransformer(
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, max_len=n_lead_days + 2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    best_epoch = 0

    # Track losses per epoch
    train_losses = []
    val_losses = []

    for epoch in range(N_EPOCHS):
        # Training
        model.train()
        epoch_train_loss = 0
        n_train_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            output = model(
                batch['input_rmm1'], batch['input_rmm2'], batch['input_amp'],
                batch['input_day_idx'], batch['attention_mask']
            )

            loss, _, _ = uar_loss(output[:, 0], output[:, 1], batch['target_rmm1'], batch['target_rmm2'], P, lam)
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                n_train_batches += 1

        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(
                    batch['input_rmm1'], batch['input_rmm2'], batch['input_amp'],
                    batch['input_day_idx'], batch['attention_mask']
                )
                loss, _, _ = uar_loss(output[:, 0], output[:, 1], batch['target_rmm1'], batch['target_rmm2'], P, lam)
                val_loss += loss.item()

        val_loss /= max(len(valid_loader), 1)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses)
    }


def evaluate_model(model, config, test_year, device):
    """Evaluate model on test set."""
    n_lead_days = config['n_lead_days']
    test_dataset = Seq2PointDataset(config['data_dir'], [test_year], n_lead_days)
    test_sampler = ForecastBatchSampler(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

    model.eval()
    all_pred1, all_pred2, all_gt1, all_gt2, all_fc1, all_fc2, all_ld = [], [], [], [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(
                batch['input_rmm1'], batch['input_rmm2'], batch['input_amp'],
                batch['input_day_idx'], batch['attention_mask']
            )

            all_pred1.extend(output[:, 0].cpu().numpy())
            all_pred2.extend(output[:, 1].cpu().numpy())
            all_gt1.extend(batch['target_rmm1'].cpu().numpy())
            all_gt2.extend(batch['target_rmm2'].cpu().numpy())
            all_fc1.extend(batch['forecast_rmm1'].cpu().numpy())
            all_fc2.extend(batch['forecast_rmm2'].cpu().numpy())
            all_ld.extend(batch['lead_day'].cpu().numpy())

    pred1, pred2 = np.array(all_pred1), np.array(all_pred2)
    gt1, gt2 = np.array(all_gt1), np.array(all_gt2)
    fc1, fc2 = np.array(all_fc1), np.array(all_fc2)
    lead_days_arr = np.array(all_ld)

    # Compute metrics PER LEAD DAY, then average across lead days
    unique_leads = np.unique(lead_days_arr)
    rmse_per_lead_model = []
    rmse_per_lead_baseline = []
    bcor_per_lead_model = []
    bcor_per_lead_baseline = []

    for lead in unique_leads:
        mask = lead_days_arr == lead
        p1, p2 = pred1[mask], pred2[mask]
        g1, g2 = gt1[mask], gt2[mask]
        f1, f2 = fc1[mask], fc2[mask]

        valid = ~(np.isnan(g1) | np.isnan(g2))
        if valid.sum() == 0:
            continue
        p1, p2 = p1[valid], p2[valid]
        g1, g2 = g1[valid], g2[valid]
        f1, f2 = f1[valid], f2[valid]

        rmse_model_ld = np.sqrt(np.mean((p1 - g1)**2 + (p2 - g2)**2))
        rmse_baseline_ld = np.sqrt(np.mean((f1 - g1)**2 + (f2 - g2)**2))
        rmse_per_lead_model.append(rmse_model_ld)
        rmse_per_lead_baseline.append(rmse_baseline_ld)

        bcor_model_ld = bcor_fn(p1, p2, g1, g2)
        bcor_baseline_ld = bcor_fn(f1, f2, g1, g2)
        bcor_per_lead_model.append(bcor_model_ld)
        bcor_per_lead_baseline.append(bcor_baseline_ld)

    avg_rmse_model = np.mean(rmse_per_lead_model)
    avg_rmse_baseline = np.mean(rmse_per_lead_baseline)
    rmse_imp = (avg_rmse_baseline - avg_rmse_model) / avg_rmse_baseline * 100

    avg_bcor_model = np.mean(bcor_per_lead_model)
    avg_bcor_baseline = np.mean(bcor_per_lead_baseline)
    bcor_imp = (avg_bcor_model - avg_bcor_baseline) / avg_bcor_baseline * 100

    valid_all = ~(np.isnan(gt1) | np.isnan(gt2))

    return {
        'rmse_imp': rmse_imp,
        'bcor_imp': bcor_imp,
        'pred_rmm1': pred1[valid_all],
        'pred_rmm2': pred2[valid_all],
        'gt_rmm1': gt1[valid_all],
        'gt_rmm2': gt2[valid_all],
        'fc_rmm1': fc1[valid_all],
        'fc_rmm2': fc2[valid_all],
        'lead_days': lead_days_arr[valid_all],
    }


def train_fold_with_lambda_selection(config, train_years, test_year, device, seed):
    """Train models with all lambda values, select best on validation (stratified by year)."""
    print(f"    Lambda selection: testing {LAMBDA_VALUES}")
    print(f"    Using stratified validation ({VAL_RATIO*100:.0f}% from each training year)")

    best_lambda = None
    best_val_loss = float('inf')
    best_model = None
    best_loss_history = None

    for lam in LAMBDA_VALUES:
        model, val_loss, loss_history = train_single_model(config, train_years, lam, device, seed)
        print(f"      lambda={lam}: val_loss={val_loss:.6f} (epochs={loss_history['total_epochs']}, best_epoch={loss_history['best_epoch']})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lambda = lam
            best_model = model
            best_loss_history = loss_history

    print(f"    Selected lambda={best_lambda} (val_loss={best_val_loss:.6f})")

    # Evaluate best model on test set
    results = evaluate_model(best_model, config, test_year, device)
    results['model'] = best_model
    results['selected_lambda'] = best_lambda
    results['loss_history'] = best_loss_history

    return results


def main(dataset, seed=DEFAULT_SEED):
    set_seed(seed)
    print("=" * 70)
    print(f"UAR METHOD: SEQ2POINT TRANSFORMER - {dataset} (seed={seed})")
    print(f"Lambda selection from: {LAMBDA_VALUES}")
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
    results_dir = os.path.join(BASE_DIR, 'baselines', 'uar', dataset, f'seed{seed}', 'results')
    model_dir = os.path.join(BASE_DIR, 'baselines', 'uar', dataset, f'seed{seed}', 'models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    all_results = []
    all_predictions = {}

    for test_year in test_years:
        print(f"\n{'=' * 60}")
        print(f"FOLD: Test Year {test_year}")

        # Training years: use full training window (stratified validation within)
        train_end = test_year - 1
        train_start = train_end - train_window + 1
        train_years = list(range(train_start, train_end + 1))  # Include all years up to train_end

        print(f"  Train/Val: {train_start}-{train_end} (stratified 85%/15%), Test: {test_year}")

        results = train_fold_with_lambda_selection(config, train_years, test_year, device, seed)

        print(f"  RMSE Improvement: {results['rmse_imp']:.2f}%")
        print(f"  BCOR Improvement: {results['bcor_imp']:.2f}%")

        # Save model
        torch.save({
            'model_state_dict': results['model'].state_dict(),
            'test_year': test_year,
        }, os.path.join(model_dir, f'model_fold_{test_year}.pt'))

        all_results.append({
            'year': test_year,
            'rmse_imp': float(results['rmse_imp']),
            'bcor_imp': float(results['bcor_imp']),
            'selected_lambda': results['selected_lambda'],
            'loss_history': results['loss_history'],
        })

        all_predictions[f'pred_rmm1_{test_year}'] = results['pred_rmm1']
        all_predictions[f'pred_rmm2_{test_year}'] = results['pred_rmm2']
        all_predictions[f'gt_rmm1_{test_year}'] = results['gt_rmm1']
        all_predictions[f'gt_rmm2_{test_year}'] = results['gt_rmm2']
        all_predictions[f'fc_rmm1_{test_year}'] = results['fc_rmm1']
        all_predictions[f'fc_rmm2_{test_year}'] = results['fc_rmm2']
        all_predictions[f'lead_days_{test_year}'] = results['lead_days']

    # Save predictions
    np.savez(os.path.join(results_dir, 'predictions.npz'), **all_predictions)

    # Summary
    rmse_imps = [r['rmse_imp'] for r in all_results]
    bcor_imps = [r['bcor_imp'] for r in all_results]
    lambda_selections = [r['selected_lambda'] for r in all_results]

    # Lambda distribution
    from collections import Counter
    lambda_counts = Counter(lambda_selections)

    summary = {
        'model': 'uar_seq2point_transformer',
        'dataset': dataset,
        'lambda_values_searched': LAMBDA_VALUES,
        'lambda_selection_counts': dict(lambda_counts),
        'rmse_improvement_mean': float(np.mean(rmse_imps)),
        'rmse_improvement_std': float(np.std(rmse_imps)),
        'bcor_improvement_mean': float(np.mean(bcor_imps)),
        'bcor_improvement_std': float(np.std(bcor_imps)),
        'per_year': all_results
    }

    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"UAR - {dataset} Results:")
    print(f"  RMSE Improvement: {np.mean(rmse_imps):.2f}% +/- {np.std(rmse_imps):.2f}%")
    print(f"  BCOR Improvement: {np.mean(bcor_imps):.2f}% +/- {np.std(bcor_imps):.2f}%")
    print(f"  Lambda selections: {dict(lambda_counts)}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['BoM', 'JMA', 'CNRM'])
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    args = parser.parse_args()
    main(args.dataset, args.seed)
