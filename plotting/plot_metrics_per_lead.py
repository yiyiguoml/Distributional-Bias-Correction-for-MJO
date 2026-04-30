"""
Plot BCOR, RMSE, and BMSE per lead day for DBC and baselines.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Auto-detect repository root from file location
BASE_PATH = Path(__file__).resolve().parent.parent

# Dataset configurations
DATASETS = {
    'BoM': {'n_lead_days': 62},
    'JMA': {'n_lead_days': 33},
    'CNRM': {'n_lead_days': 60},
}

# Method configurations (color, linestyle, linewidth)
METHODS = {
    'Raw': {'color': 'gray', 'linestyle': '--', 'linewidth': 2, 'marker': None},
    'DBC': {'color': '#e41a1c', 'linestyle': '-', 'linewidth': 2.5, 'marker': None},  # Red
    'Silini': {'color': '#377eb8', 'linestyle': '-', 'linewidth': 1.5, 'marker': None},  # Blue
    'UAR': {'color': '#4daf4a', 'linestyle': '-', 'linewidth': 1.5, 'marker': None},  # Green
    'Kim': {'color': '#984ea3', 'linestyle': '-', 'linewidth': 1.5, 'marker': None},  # Purple
    'EMOS': {'color': '#ff7f00', 'linestyle': ':', 'linewidth': 1.5, 'marker': None},  # Orange
    'BMA': {'color': '#a65628', 'linestyle': ':', 'linewidth': 1.5, 'marker': None},  # Brown
    'BNN': {'color': '#f781bf', 'linestyle': ':', 'linewidth': 1.5, 'marker': None},  # Pink
}


def bcor_fn(p1, p2, g1, g2):
    """Bivariate correlation (handles NaN)."""
    # Remove NaN values
    valid = ~(np.isnan(p1) | np.isnan(p2) | np.isnan(g1) | np.isnan(g2))
    if valid.sum() == 0:
        return np.nan
    p1, p2, g1, g2 = p1[valid], p2[valid], g1[valid], g2[valid]
    num = (p1 * g1 + p2 * g2).sum()
    den = np.sqrt((p1**2 + p2**2).sum() * (g1**2 + g2**2).sum()) + 1e-8
    return num / den


def rmse_fn(p1, p2, g1, g2):
    """Bivariate RMSE (handles NaN)."""
    valid = ~(np.isnan(p1) | np.isnan(p2) | np.isnan(g1) | np.isnan(g2))
    if valid.sum() == 0:
        return np.nan
    p1, p2, g1, g2 = p1[valid], p2[valid], g1[valid], g2[valid]
    return np.sqrt(np.mean((p1 - g1)**2 + (p2 - g2)**2))


def bmse_fn(p1, p2, g1, g2):
    """Bivariate MSE (handles NaN)."""
    valid = ~(np.isnan(p1) | np.isnan(p2) | np.isnan(g1) | np.isnan(g2))
    if valid.sum() == 0:
        return np.nan
    p1, p2, g1, g2 = p1[valid], p2[valid], g1[valid], g2[valid]
    return np.mean((p1 - g1)**2 + (p2 - g2)**2)


def compute_metrics_per_lead(pred1, pred2, gt1, gt2, leads, n_lead_days):
    """Compute metrics per lead day."""
    lead_days = np.arange(1, n_lead_days + 1)
    bcor = np.zeros(n_lead_days)
    rmse = np.zeros(n_lead_days)
    bmse = np.zeros(n_lead_days)

    for i, ld in enumerate(lead_days):
        mask = leads == ld
        if mask.sum() == 0:
            bcor[i] = np.nan
            rmse[i] = np.nan
            bmse[i] = np.nan
            continue
        bcor[i] = bcor_fn(pred1[mask], pred2[mask], gt1[mask], gt2[mask])
        rmse[i] = rmse_fn(pred1[mask], pred2[mask], gt1[mask], gt2[mask])
        bmse[i] = bmse_fn(pred1[mask], pred2[mask], gt1[mask], gt2[mask])

    return {'bcor': bcor, 'rmse': rmse, 'bmse': bmse, 'lead_days': lead_days}


def load_all_methods(dataset):
    """Load predictions for all methods."""
    n_lead_days = DATASETS[dataset]['n_lead_days']
    results = {}

    # Load DBC predictions
    dbc_path = BASE_PATH / f"{dataset}_norm/seed42/results/predictions.npz"
    if dbc_path.exists():
        data = np.load(dbc_path)
        years = sorted(set(k.split('_')[-1] for k in data.keys() if k.startswith('gt_rmm1')))

        all_gt1, all_gt2, all_fc1, all_fc2, all_dbc1, all_dbc2, all_leads = [], [], [], [], [], [], []
        for year in years:
            all_gt1.append(data[f'gt_rmm1_{year}'])
            all_gt2.append(data[f'gt_rmm2_{year}'])
            all_fc1.append(data[f'forecast_rmm1_{year}'])
            all_fc2.append(data[f'forecast_rmm2_{year}'])
            all_dbc1.append(data[f'quantiles_rmm1_{year}'][:, 3])  # Median
            all_dbc2.append(data[f'quantiles_rmm2_{year}'][:, 3])
            all_leads.append(data[f'lead_days_{year}'])

        gt1 = np.concatenate(all_gt1)
        gt2 = np.concatenate(all_gt2)
        fc1 = np.concatenate(all_fc1)
        fc2 = np.concatenate(all_fc2)
        dbc1 = np.concatenate(all_dbc1)
        dbc2 = np.concatenate(all_dbc2)
        leads = np.concatenate(all_leads)

        results['Raw'] = compute_metrics_per_lead(fc1, fc2, gt1, gt2, leads, n_lead_days)
        results['DBC'] = compute_metrics_per_lead(dbc1, dbc2, gt1, gt2, leads, n_lead_days)

    # Load deterministic baselines (Kim, Silini, UAR)
    for method in ['kim', 'silini', 'uar']:
        path = BASE_PATH / f"baselines/{method}/{dataset}/seed42/results/predictions.npz"
        if path.exists():
            data = np.load(path)
            years = sorted(set(k.split('_')[-1] for k in data.keys() if k.startswith('gt_rmm1')))

            all_pred1, all_pred2, all_gt1, all_gt2 = [], [], [], []
            for year in years:
                all_pred1.append(data[f'pred_rmm1_{year}'])
                all_pred2.append(data[f'pred_rmm2_{year}'])
                all_gt1.append(data[f'gt_rmm1_{year}'])
                all_gt2.append(data[f'gt_rmm2_{year}'])

            pred1 = np.concatenate(all_pred1, axis=0)
            pred2 = np.concatenate(all_pred2, axis=0)
            gt1 = np.concatenate(all_gt1, axis=0)
            gt2 = np.concatenate(all_gt2, axis=0)

            # Handle both 2D (kim, silini) and 1D (uar) formats
            if pred1.ndim == 2:
                # Shape is (n_forecasts, n_lead_days)
                n_forecasts, actual_n_leads = pred1.shape
                leads = np.tile(np.arange(1, actual_n_leads + 1), n_forecasts)
                pred1 = pred1.flatten()
                pred2 = pred2.flatten()
                gt1 = gt1.flatten()
                gt2 = gt2.flatten()
            else:
                # Flat array - need to get lead_days from file
                all_leads = []
                for year in years:
                    all_leads.append(data[f'lead_days_{year}'])
                leads = np.concatenate(all_leads)

            results[method.capitalize()] = compute_metrics_per_lead(pred1, pred2, gt1, gt2, leads, n_lead_days)

    # Load probabilistic baselines (EMOS, BMA, BNN) - use median, shape is (n_forecasts, n_lead_days, 7)
    prob_names = {'emos': 'EMOS', 'bma': 'BMA', 'bnn_mlp': 'BNN'}
    for method, display_name in prob_names.items():
        path = BASE_PATH / f"baselines/probabilistic/{dataset}/results/{method}_predictions.npz"
        if path.exists():
            data = np.load(path)
            years = sorted(set(k.split('_')[-1] for k in data.keys() if k.startswith('gt_rmm1')))

            all_pred1, all_pred2, all_gt1, all_gt2 = [], [], [], []
            for year in years:
                # Use median (index 3) from quantiles - shape (n_forecasts, n_lead_days, 7)
                all_pred1.append(data[f'quantiles_rmm1_{year}'][:, :, 3])  # (n_forecasts, n_lead_days)
                all_pred2.append(data[f'quantiles_rmm2_{year}'][:, :, 3])
                all_gt1.append(data[f'gt_rmm1_{year}'])
                all_gt2.append(data[f'gt_rmm2_{year}'])

            pred1 = np.concatenate(all_pred1, axis=0)
            pred2 = np.concatenate(all_pred2, axis=0)
            gt1 = np.concatenate(all_gt1, axis=0)
            gt2 = np.concatenate(all_gt2, axis=0)

            # Flatten and create lead days array
            n_forecasts, actual_n_leads = pred1.shape
            leads = np.tile(np.arange(1, actual_n_leads + 1), n_forecasts)
            pred1 = pred1.flatten()
            pred2 = pred2.flatten()
            gt1 = gt1.flatten()
            gt2 = gt2.flatten()

            results[display_name] = compute_metrics_per_lead(pred1, pred2, gt1, gt2, leads, n_lead_days)

    return results


def main():
    metrics = ['bcor', 'rmse', 'bmse']
    titles = ['BCOR (higher is better)', 'RMSE (lower is better)', 'BMSE (lower is better)']
    dataset_list = list(DATASETS.keys())

    # Load data for all datasets
    all_results = {}
    for dataset in dataset_list:
        print(f"Loading {dataset}...")
        all_results[dataset] = load_all_methods(dataset)

    # Create 3x3 figure (rows=datasets, cols=metrics)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for row, dataset in enumerate(dataset_list):
        results = all_results[dataset]
        n_lead_days = DATASETS[dataset]['n_lead_days']
        lead_days = np.arange(1, n_lead_days + 1)

        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            # Plot each method
            for method_name, cfg in METHODS.items():
                if method_name in results:
                    ax.plot(lead_days, results[method_name][metric],
                            color=cfg['color'],
                            linestyle=cfg['linestyle'],
                            linewidth=cfg['linewidth'],
                            label=method_name)

            ax.set_xlabel('Lead Day', fontsize=11)
            ax.grid(True, alpha=0.3)

            # Y-axis label with dataset name on leftmost column
            if col == 0:
                ax.set_ylabel(f'{dataset}', fontsize=12, fontweight='bold')

            # Column titles on top row only
            if row == 0:
                ax.set_title(titles[col], fontsize=13, fontweight='bold')

            # Legend only in first subplot
            if row == 0 and col == 0:
                ax.legend(loc='lower left', fontsize=9, ncol=2)

    plt.tight_layout()

    # Save figure (PDF only)
    output_pdf = BASE_PATH / "plot/metrics_per_lead.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved: {output_pdf}")

    plt.close()


if __name__ == "__main__":
    main()
