"""
Dataset and data loading utilities for unified DBC training.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class WalkForwardDataset(Dataset):
    """
    Dataset for walk-forward validation with variable lead days.
    """
    def __init__(self, data_dir, years, n_lead_days):
        """
        Args:
            data_dir: Path to data directory
            years: List of years to include
            n_lead_days: Number of lead days for this dataset
        """
        self.n_lead_days = n_lead_days

        # Load data
        self.forecast_rmm1 = np.load(os.path.join(data_dir, 'forecast_rmm1_with_day0.npy'))
        self.forecast_rmm2 = np.load(os.path.join(data_dir, 'forecast_rmm2_with_day0.npy'))
        self.gt_rmm1 = np.load(os.path.join(data_dir, 'ground_truth_rmm1.npy'))
        self.gt_rmm2 = np.load(os.path.join(data_dir, 'ground_truth_rmm2.npy'))
        self.amplitude_forecast = np.load(os.path.join(data_dir, 'amplitude_forecast.npy'))
        self.day0_info = np.load(os.path.join(data_dir, 'day0_info.npy'))
        self.times = np.load(os.path.join(data_dir, 'times.npy'), allow_pickle=True)

        # Get years
        self.datetimes = pd.to_datetime(self.times)
        self.years_array = self.datetimes.year.values

        # Filter by years
        mask = np.isin(self.years_array, years)
        self.forecast_rmm1 = self.forecast_rmm1[mask]
        self.forecast_rmm2 = self.forecast_rmm2[mask]
        self.gt_rmm1 = self.gt_rmm1[mask]
        self.gt_rmm2 = self.gt_rmm2[mask]
        self.amplitude_forecast = self.amplitude_forecast[mask]
        self.day0_info = self.day0_info[mask]
        self.datetimes = self.datetimes[mask]

        # Extract day0 amplitude (column 3 after our data fix)
        self.day0_amplitude = self.day0_info[:, 3]

        # Create full amplitude sequence (day0 + forecast)
        self.forecast_amplitude_full = np.zeros((len(self.forecast_rmm1), n_lead_days + 1))
        self.forecast_amplitude_full[:, 0] = self.day0_amplitude
        self.forecast_amplitude_full[:, 1:] = self.amplitude_forecast[:, :n_lead_days]

        # Create samples: (forecast_idx, lead_day)
        # Filter out samples with NaN targets
        self.n_forecasts = len(self.forecast_rmm1)
        self.samples = []
        for fc in range(self.n_forecasts):
            for ld in range(1, n_lead_days + 1):
                gt1 = self.gt_rmm1[fc, ld - 1]
                gt2 = self.gt_rmm2[fc, ld - 1]
                if not (np.isnan(gt1) or np.isnan(gt2)):
                    self.samples.append((fc, ld))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fc_idx, lead_day = self.samples[idx]
        seq_len = lead_day + 1  # Include day0

        return {
            'input_rmm1': torch.FloatTensor(self.forecast_rmm1[fc_idx, :seq_len]),
            'input_rmm2': torch.FloatTensor(self.forecast_rmm2[fc_idx, :seq_len]),
            'input_amp': torch.FloatTensor(self.forecast_amplitude_full[fc_idx, :seq_len]),
            'target_rmm1': torch.FloatTensor([self.gt_rmm1[fc_idx, lead_day - 1]]),
            'target_rmm2': torch.FloatTensor([self.gt_rmm2[fc_idx, lead_day - 1]]),
            'forecast_rmm1': torch.FloatTensor([self.forecast_rmm1[fc_idx, lead_day]]),
            'forecast_rmm2': torch.FloatTensor([self.forecast_rmm2[fc_idx, lead_day]]),
            'lead_day': torch.LongTensor([lead_day]),
        }


class ForecastBatchSampler(Sampler):
    """
    Sampler that groups all lead days for each forecast together.
    Works with filtered samples (handles NaN-filtered datasets).
    """
    def __init__(self, dataset, shuffle=True):
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
        for fc_idx in order:
            yield self.forecast_to_indices[fc_idx]

    def __len__(self):
        return len(self.forecast_order)


def collate_fn(batch):
    """
    Collate function for variable-length sequences.
    Pads sequences and creates attention masks.
    """
    return {
        'input_rmm1': torch.nn.utils.rnn.pad_sequence(
            [b['input_rmm1'] for b in batch], batch_first=True),
        'input_rmm2': torch.nn.utils.rnn.pad_sequence(
            [b['input_rmm2'] for b in batch], batch_first=True),
        'input_amp': torch.nn.utils.rnn.pad_sequence(
            [b['input_amp'] for b in batch], batch_first=True),
        'attention_mask': torch.nn.utils.rnn.pad_sequence(
            [torch.ones(len(b['input_rmm1'])) for b in batch],
            batch_first=True, padding_value=0),
        'target_rmm1': torch.cat([b['target_rmm1'] for b in batch]),
        'target_rmm2': torch.cat([b['target_rmm2'] for b in batch]),
        'forecast_rmm1': torch.cat([b['forecast_rmm1'] for b in batch]),
        'forecast_rmm2': torch.cat([b['forecast_rmm2'] for b in batch]),
        'lead_day': torch.cat([b['lead_day'] for b in batch]),
    }


def compute_baseline_rmse_per_lead(dataset):
    """
    Compute baseline RMSE per lead day from the dataset.
    Used for loss normalization.

    Returns:
        numpy array of shape (n_lead_days,) with RMSE for each lead day
    """
    n_lead_days = dataset.n_lead_days
    rmse_per_lead = np.zeros(n_lead_days)
    counts_per_lead = np.zeros(n_lead_days)

    for fc_idx in range(dataset.n_forecasts):
        for lead_day in range(1, n_lead_days + 1):
            gt1 = dataset.gt_rmm1[fc_idx, lead_day - 1]
            gt2 = dataset.gt_rmm2[fc_idx, lead_day - 1]
            fc1 = dataset.forecast_rmm1[fc_idx, lead_day]
            fc2 = dataset.forecast_rmm2[fc_idx, lead_day]

            # Skip NaN values
            if np.isnan(gt1) or np.isnan(gt2):
                continue

            # Bivariate RMSE
            rmse = np.sqrt(((fc1 - gt1)**2 + (fc2 - gt2)**2) / 2)
            rmse_per_lead[lead_day - 1] += rmse
            counts_per_lead[lead_day - 1] += 1

    rmse_per_lead = rmse_per_lead / (counts_per_lead + 1e-8)
    return rmse_per_lead
