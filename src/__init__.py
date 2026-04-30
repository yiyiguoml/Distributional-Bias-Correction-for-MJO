"""
MJO Probabilistic Correction with Deep Bias Correction (DBC)

This module contains the core implementation of the DBC Transformer model
for probabilistic bias correction of MJO (Madden-Julian Oscillation) forecasts.
"""

from .model import DBCTransformerNoMeta, NonCrossingQuantileHead
from .loss import (
    pinball_loss,
    bivariate_dbc_loss,
    bivariate_dbc_loss_normalized,
    compute_crps_from_quantiles
)
from .config import DATASETS, QUANTILE_LEVELS, set_seed
from .dataset import WalkForwardDataset, ForecastBatchSampler, collate_fn

__version__ = "1.0.0"
__all__ = [
    'DBCTransformerNoMeta',
    'NonCrossingQuantileHead',
    'pinball_loss',
    'bivariate_dbc_loss',
    'bivariate_dbc_loss_normalized',
    'compute_crps_from_quantiles',
    'DATASETS',
    'QUANTILE_LEVELS',
    'set_seed',
    'WalkForwardDataset',
    'ForecastBatchSampler',
    'collate_fn',
]
