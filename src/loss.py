"""
Loss functions for DBC (Deep Bias Correction) training.
"""

import torch


def pinball_loss(y_true, y_pred, tau):
    """
    Standard pinball (quantile) loss without normalization.

    Args:
        y_true: (B,) true values
        y_pred: (B,) predicted quantiles
        tau: quantile level (0-1)

    Returns:
        scalar loss
    """
    error = y_true - y_pred
    loss = torch.max(tau * error, (tau - 1) * error)
    return torch.mean(loss)


def pinball_loss_normalized(y_true, y_pred, tau, normalization):
    """
    Pinball loss normalized by baseline RMSE.
    Ensures each lead day contributes equally regardless of error magnitude.

    Args:
        y_true: (B,) true values
        y_pred: (B,) predicted quantiles
        tau: quantile level (0-1)
        normalization: (B,) normalization factor (baseline RMSE per sample)

    Returns:
        scalar loss
    """
    error = y_true - y_pred
    loss = torch.max(tau * error, (tau - 1) * error)
    normalized_loss = loss / (normalization + 1e-8)
    return torch.mean(normalized_loss)


def bivariate_dbc_loss(quantile_predictions, target_rmm1, target_rmm2,
                       quantile_levels):
    """
    Combined pinball loss for bivariate DBC WITHOUT normalization.

    Args:
        quantile_predictions: (B, 2, n_quantiles) predicted quantiles
        target_rmm1: (B,) true RMM1 values
        target_rmm2: (B,) true RMM2 values
        quantile_levels: list of quantile levels

    Returns:
        scalar loss
    """
    n_quantiles = len(quantile_levels)

    pred_rmm1 = quantile_predictions[:, 0, :]  # (B, n_quantiles)
    pred_rmm2 = quantile_predictions[:, 1, :]

    total_loss = 0
    for k, tau in enumerate(quantile_levels):
        tau_tensor = torch.tensor(tau, device=quantile_predictions.device)
        total_loss += pinball_loss(target_rmm1, pred_rmm1[:, k], tau_tensor)
        total_loss += pinball_loss(target_rmm2, pred_rmm2[:, k], tau_tensor)

    return total_loss / (2 * n_quantiles)


def bivariate_dbc_loss_normalized(quantile_predictions, target_rmm1, target_rmm2,
                                   quantile_levels, lead_days, baseline_rmse_per_lead):
    """
    Combined pinball loss for bivariate DBC WITH normalization.

    Args:
        quantile_predictions: (B, 2, n_quantiles) predicted quantiles
        target_rmm1: (B,) true RMM1 values
        target_rmm2: (B,) true RMM2 values
        quantile_levels: list of quantile levels
        lead_days: (B,) lead day for each sample
        baseline_rmse_per_lead: tensor of baseline RMSE per lead day

    Returns:
        scalar loss
    """
    n_quantiles = len(quantile_levels)

    # Get normalization factor for each sample
    norm_factors = baseline_rmse_per_lead[lead_days - 1]  # (B,)

    pred_rmm1 = quantile_predictions[:, 0, :]  # (B, n_quantiles)
    pred_rmm2 = quantile_predictions[:, 1, :]

    total_loss = 0
    for k, tau in enumerate(quantile_levels):
        tau_tensor = torch.tensor(tau, device=quantile_predictions.device)
        total_loss += pinball_loss_normalized(
            target_rmm1, pred_rmm1[:, k], tau_tensor, norm_factors)
        total_loss += pinball_loss_normalized(
            target_rmm2, pred_rmm2[:, k], tau_tensor, norm_factors)

    return total_loss / (2 * n_quantiles)


def bcor_loss(pred_rmm1, pred_rmm2, target_rmm1, target_rmm2):
    """
    BCOR-based loss to maximize bivariate correlation.

    Args:
        pred_rmm1: (B,) predicted RMM1 (median quantile)
        pred_rmm2: (B,) predicted RMM2 (median quantile)
        target_rmm1: (B,) true RMM1
        target_rmm2: (B,) true RMM2

    Returns:
        scalar loss (1 - BCOR), minimize to maximize BCOR
    """
    # BCOR = sum(p1*g1 + p2*g2) / sqrt(sum(p1^2+p2^2) * sum(g1^2+g2^2))
    num = (pred_rmm1 * target_rmm1 + pred_rmm2 * target_rmm2).sum()
    den = torch.sqrt(
        (pred_rmm1**2 + pred_rmm2**2).sum() *
        (target_rmm1**2 + target_rmm2**2).sum()
    ) + 1e-8
    bcor = num / den
    return 1 - bcor


def bivariate_dbc_loss_with_bcor(quantile_predictions, target_rmm1, target_rmm2,
                                  quantile_levels, lambda_bcor=0.1):
    """
    Combined pinball loss + BCOR loss for bivariate DBC.

    Args:
        quantile_predictions: (B, 2, n_quantiles) predicted quantiles
        target_rmm1: (B,) true RMM1 values
        target_rmm2: (B,) true RMM2 values
        quantile_levels: list of quantile levels
        lambda_bcor: weight for BCOR loss term

    Returns:
        total_loss, pinball_loss, bcor_loss_value
    """
    n_quantiles = len(quantile_levels)
    median_idx = quantile_levels.index(0.5)

    pred_rmm1 = quantile_predictions[:, 0, :]  # (B, n_quantiles)
    pred_rmm2 = quantile_predictions[:, 1, :]

    # Pinball loss (standard)
    pinball_total = 0
    for k, tau in enumerate(quantile_levels):
        tau_tensor = torch.tensor(tau, device=quantile_predictions.device)
        pinball_total += pinball_loss(target_rmm1, pred_rmm1[:, k], tau_tensor)
        pinball_total += pinball_loss(target_rmm2, pred_rmm2[:, k], tau_tensor)
    pinball_total = pinball_total / (2 * n_quantiles)

    # BCOR loss (on median predictions)
    pred_median_rmm1 = pred_rmm1[:, median_idx]
    pred_median_rmm2 = pred_rmm2[:, median_idx]
    bcor_loss_val = bcor_loss(pred_median_rmm1, pred_median_rmm2, target_rmm1, target_rmm2)

    # Combined loss
    total_loss = pinball_total + lambda_bcor * bcor_loss_val

    return total_loss, pinball_total, bcor_loss_val


def compute_crps_from_quantiles(quantiles, y_true, quantile_levels):
    """
    Compute CRPS from quantile predictions using pinball loss.

    Args:
        quantiles: (N, n_quantiles) predicted quantiles
        y_true: (N,) true values
        quantile_levels: list of quantile levels

    Returns:
        scalar CRPS
    """
    import numpy as np

    total_pinball = 0.0
    for k, tau in enumerate(quantile_levels):
        error = y_true - quantiles[:, k]
        pinball = np.where(error >= 0, tau * error, (tau - 1) * error)
        total_pinball += pinball.mean()

    # Scale to approximate CRPS
    return 2 * total_pinball / len(quantile_levels)
