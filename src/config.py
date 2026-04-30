"""
Configuration for unified DBC training across datasets.
"""

import os
import random
import numpy as np
import torch

# Reproducibility
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

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset configurations
DATASETS = {
    'bom': {
        'name': 'BoM',
        'data_dir': os.path.join(BASE_DIR, 'data/BOM'),
        'n_lead_days': 62,
        'max_seq_len': 63,
        'all_years': list(range(1981, 2014)),
        'test_years': list(range(1999, 2014)),
    },
    'jma': {
        'name': 'JMA',
        'data_dir': os.path.join(BASE_DIR, 'data/JMA'),
        'n_lead_days': 33,
        'max_seq_len': 34,
        'all_years': list(range(1981, 2013)),
        'test_years': list(range(1998, 2013)),
    },
    'cnrm': {
        'name': 'CNRM',
        'data_dir': os.path.join(BASE_DIR, 'data/CNRM'),
        'n_lead_days': 60,
        'max_seq_len': 61,
        'all_years': list(range(1993, 2015)),
        'test_years': list(range(2010, 2015)),
    },
}

# Quantile levels for DBC
QUANTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
N_QUANTILES = len(QUANTILE_LEVELS)

# Model architecture
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# Training configuration
N_EPOCHS = 60
PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_WINDOW = 15  # Maximum years in training set


def get_output_dir(dataset_name, normalize):
    """Get output directory for a dataset and normalization setting."""
    suffix = 'norm' if normalize else 'no_norm'
    return os.path.join(BASE_DIR, f'{DATASETS[dataset_name]["name"]}_{suffix}')
