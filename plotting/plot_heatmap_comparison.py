"""
Heatmap visualization comparing all methods across datasets and lead day ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Auto-detect repository root from file location
BASE_PATH = Path(__file__).resolve().parent.parent

# Dataset configurations
DATASETS = ['BoM', 'JMA', 'CNRM']
DATASET_LEADS = {'BoM': 62, 'JMA': 33, 'CNRM': 60}

# Methods to compare (excluding Raw which is baseline)
METHODS = ['DBC', 'Silini', 'UAR', 'Kim', 'EMOS', 'BMA', 'BNN']

# Lead day ranges
LEAD_RANGES = {
    'Short (1-15)': (1, 15),
    'Medium (16-30)': (16, 30),
    'Long (31+)': (31, None),  # None means max available
}


def bcor_fn(p1, p2, g1, g2):
    """Bivariate correlation (handles NaN)."""
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


def load_method_data(dataset, method):
    """Load prediction data for a method."""
    n_lead_days = DATASET_LEADS[dataset]

    if method == 'DBC' or method == 'Raw':
        path = BASE_PATH / f"{dataset}_norm/seed42/results/predictions.npz"
        if not path.exists():
            return None
        data = np.load(path)
        years = sorted(set(k.split('_')[-1] for k in data.keys() if k.startswith('gt_rmm1')))

        all_pred1, all_pred2, all_gt1, all_gt2, all_leads = [], [], [], [], []
        for year in years:
            all_gt1.append(data[f'gt_rmm1_{year}'])
            all_gt2.append(data[f'gt_rmm2_{year}'])
            all_leads.append(data[f'lead_days_{year}'])
            if method == 'DBC':
                all_pred1.append(data[f'quantiles_rmm1_{year}'][:, 3])
                all_pred2.append(data[f'quantiles_rmm2_{year}'][:, 3])
            else:  # Raw
                all_pred1.append(data[f'forecast_rmm1_{year}'])
                all_pred2.append(data[f'forecast_rmm2_{year}'])

        return {
            'pred1': np.concatenate(all_pred1),
            'pred2': np.concatenate(all_pred2),
            'gt1': np.concatenate(all_gt1),
            'gt2': np.concatenate(all_gt2),
            'leads': np.concatenate(all_leads),
        }

    elif method in ['Kim', 'Silini', 'UAR']:
        path = BASE_PATH / f"baselines/{method.lower()}/{dataset}/seed42/results/predictions.npz"
        if not path.exists():
            return None
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

        if pred1.ndim == 2:
            n_forecasts, actual_n_leads = pred1.shape
            leads = np.tile(np.arange(1, actual_n_leads + 1), n_forecasts)
            pred1 = pred1.flatten()
            pred2 = pred2.flatten()
            gt1 = gt1.flatten()
            gt2 = gt2.flatten()
        else:
            all_leads = []
            for year in years:
                all_leads.append(data[f'lead_days_{year}'])
            leads = np.concatenate(all_leads)

        return {'pred1': pred1, 'pred2': pred2, 'gt1': gt1, 'gt2': gt2, 'leads': leads}

    elif method in ['EMOS', 'BMA', 'BNN']:
        method_file = {'EMOS': 'emos', 'BMA': 'bma', 'BNN': 'bnn_mlp'}[method]
        path = BASE_PATH / f"baselines/probabilistic/{dataset}/results/{method_file}_predictions.npz"
        if not path.exists():
            return None
        data = np.load(path)
        years = sorted(set(k.split('_')[-1] for k in data.keys() if k.startswith('gt_rmm1')))

        all_pred1, all_pred2, all_gt1, all_gt2 = [], [], [], []
        for year in years:
            all_pred1.append(data[f'quantiles_rmm1_{year}'][:, :, 3])
            all_pred2.append(data[f'quantiles_rmm2_{year}'][:, :, 3])
            all_gt1.append(data[f'gt_rmm1_{year}'])
            all_gt2.append(data[f'gt_rmm2_{year}'])

        pred1 = np.concatenate(all_pred1, axis=0)
        pred2 = np.concatenate(all_pred2, axis=0)
        gt1 = np.concatenate(all_gt1, axis=0)
        gt2 = np.concatenate(all_gt2, axis=0)

        n_forecasts, actual_n_leads = pred1.shape
        leads = np.tile(np.arange(1, actual_n_leads + 1), n_forecasts)
        pred1 = pred1.flatten()
        pred2 = pred2.flatten()
        gt1 = gt1.flatten()
        gt2 = gt2.flatten()

        return {'pred1': pred1, 'pred2': pred2, 'gt1': gt1, 'gt2': gt2, 'leads': leads}

    return None


def compute_metrics_for_range(data, lead_min, lead_max):
    """Compute metrics for a specific lead day range."""
    if data is None:
        return {'bcor': np.nan, 'rmse': np.nan, 'bmse': np.nan}

    if lead_max is None:
        lead_max = data['leads'].max()

    mask = (data['leads'] >= lead_min) & (data['leads'] <= lead_max)
    if mask.sum() == 0:
        return {'bcor': np.nan, 'rmse': np.nan, 'bmse': np.nan}

    p1, p2 = data['pred1'][mask], data['pred2'][mask]
    g1, g2 = data['gt1'][mask], data['gt2'][mask]

    return {
        'bcor': bcor_fn(p1, p2, g1, g2),
        'rmse': rmse_fn(p1, p2, g1, g2),
        'bmse': bmse_fn(p1, p2, g1, g2),
    }


def main():
    print("Loading data for all methods and datasets...")

    # Compute metrics for all combinations
    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        # Load Raw baseline
        raw_data = load_method_data(dataset, 'Raw')
        results[dataset]['Raw'] = {}
        for range_name, (lead_min, lead_max) in LEAD_RANGES.items():
            results[dataset]['Raw'][range_name] = compute_metrics_for_range(raw_data, lead_min, lead_max)

        # Load other methods
        for method in METHODS:
            print(f"  Loading {method} for {dataset}...")
            method_data = load_method_data(dataset, method)
            results[dataset][method] = {}
            for range_name, (lead_min, lead_max) in LEAD_RANGES.items():
                results[dataset][method][range_name] = compute_metrics_for_range(method_data, lead_min, lead_max)

    # Compute improvement percentages
    metrics = ['bcor', 'rmse', 'bmse']
    range_names = list(LEAD_RANGES.keys())

    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    for row, range_name in enumerate(range_names):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            # Build matrix: methods × datasets
            matrix = np.zeros((len(METHODS), len(DATASETS)))
            for i, method in enumerate(METHODS):
                for j, dataset in enumerate(DATASETS):
                    raw_val = results[dataset]['Raw'][range_name][metric]
                    method_val = results[dataset][method][range_name][metric]

                    if np.isnan(raw_val) or np.isnan(method_val):
                        matrix[i, j] = np.nan
                    elif metric == 'bcor':
                        # Higher is better
                        matrix[i, j] = (method_val - raw_val) / raw_val * 100
                    else:
                        # Lower is better (RMSE, BMSE)
                        matrix[i, j] = (raw_val - method_val) / raw_val * 100

            # Create heatmap
            sns.heatmap(matrix, ax=ax, cmap='RdYlGn', center=0,
                        annot=True, fmt='.1f',
                        xticklabels=DATASETS, yticklabels=METHODS if col == 0 else False,
                        cbar=col == 2,  # Only show colorbar on right
                        vmin=-20, vmax=20)

            # Labels
            if row == 0:
                ax.set_title(f'{metric.upper()} Improvement %', fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(range_name, fontsize=11, fontweight='bold')

    plt.suptitle('Method Comparison: Improvement over Raw Forecast (%)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_path = BASE_PATH / "plot/heatmap_comparison.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
