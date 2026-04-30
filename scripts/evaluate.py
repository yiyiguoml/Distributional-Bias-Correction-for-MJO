#!/usr/bin/env python3
"""
Compute metrics for ALL methods using identical computation.
Metrics: RMSE, BCOR, BMSE (per-lead-day averaged)
"""

import os
import json
import numpy as np

# Auto-detect repository root from file location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIG = {
    'BoM': {'n_lead_days': 62, 'test_years': list(range(1999, 2014))},
    'JMA': {'n_lead_days': 33, 'test_years': list(range(1998, 2013))},
    'CNRM': {'n_lead_days': 60, 'test_years': list(range(2010, 2015))},
}


def bcor_fn(p1, p2, g1, g2):
    """Bivariate correlation."""
    num = (p1 * g1 + p2 * g2).sum()
    den = np.sqrt((p1**2 + p2**2).sum() * (g1**2 + g2**2).sum()) + 1e-8
    return num / den


def rmse_fn(p1, p2, g1, g2):
    """Bivariate RMSE."""
    return np.sqrt(np.mean((p1 - g1)**2 + (p2 - g2)**2))


def bmse_fn(p1, p2, g1, g2):
    """Bivariate MSE."""
    return np.mean((p1 - g1)**2 + (p2 - g2)**2)


def compute_metrics_per_lead_avg(pred1, pred2, gt1, gt2, fc1, fc2, n_lead_days):
    """
    Compute metrics using per-lead-day averaging.

    Returns metrics for both model predictions and raw forecast baseline.
    """
    # Flatten if needed
    pred1 = pred1.flatten()
    pred2 = pred2.flatten()
    gt1 = gt1.flatten()
    gt2 = gt2.flatten()
    fc1 = fc1.flatten()
    fc2 = fc2.flatten()

    # Remove NaN
    valid = ~(np.isnan(pred1) | np.isnan(pred2) | np.isnan(gt1) | np.isnan(gt2))
    pred1, pred2 = pred1[valid], pred2[valid]
    gt1, gt2 = gt1[valid], gt2[valid]
    fc1, fc2 = fc1[valid], fc2[valid]

    n_samples = len(pred1)
    n_forecasts = n_samples // n_lead_days

    if n_forecasts * n_lead_days != n_samples:
        # Fallback to overall if doesn't reshape cleanly
        return {
            'rmse_model': rmse_fn(pred1, pred2, gt1, gt2),
            'rmse_baseline': rmse_fn(fc1, fc2, gt1, gt2),
            'bcor_model': bcor_fn(pred1, pred2, gt1, gt2),
            'bcor_baseline': bcor_fn(fc1, fc2, gt1, gt2),
            'bmse_model': bmse_fn(pred1, pred2, gt1, gt2),
            'bmse_baseline': bmse_fn(fc1, fc2, gt1, gt2),
        }

    # Reshape to (n_forecasts, n_lead_days)
    pred1_2d = pred1.reshape(n_forecasts, n_lead_days)
    pred2_2d = pred2.reshape(n_forecasts, n_lead_days)
    gt1_2d = gt1.reshape(n_forecasts, n_lead_days)
    gt2_2d = gt2.reshape(n_forecasts, n_lead_days)
    fc1_2d = fc1.reshape(n_forecasts, n_lead_days)
    fc2_2d = fc2.reshape(n_forecasts, n_lead_days)

    # Compute per lead day
    rmse_model_per_lead = []
    rmse_baseline_per_lead = []
    bcor_model_per_lead = []
    bcor_baseline_per_lead = []
    bmse_model_per_lead = []
    bmse_baseline_per_lead = []

    for ld in range(n_lead_days):
        p1, p2 = pred1_2d[:, ld], pred2_2d[:, ld]
        g1, g2 = gt1_2d[:, ld], gt2_2d[:, ld]
        f1, f2 = fc1_2d[:, ld], fc2_2d[:, ld]

        rmse_model_per_lead.append(rmse_fn(p1, p2, g1, g2))
        rmse_baseline_per_lead.append(rmse_fn(f1, f2, g1, g2))
        bcor_model_per_lead.append(bcor_fn(p1, p2, g1, g2))
        bcor_baseline_per_lead.append(bcor_fn(f1, f2, g1, g2))
        bmse_model_per_lead.append(bmse_fn(p1, p2, g1, g2))
        bmse_baseline_per_lead.append(bmse_fn(f1, f2, g1, g2))

    return {
        'rmse_model': np.mean(rmse_model_per_lead),
        'rmse_baseline': np.mean(rmse_baseline_per_lead),
        'bcor_model': np.mean(bcor_model_per_lead),
        'bcor_baseline': np.mean(bcor_baseline_per_lead),
        'bmse_model': np.mean(bmse_model_per_lead),
        'bmse_baseline': np.mean(bmse_baseline_per_lead),
    }


def load_and_eval_dbc(dataset, variant='norm', seed=42):
    """Load DBC predictions and compute metrics."""
    config = DATASET_CONFIG[dataset]
    test_years = config['test_years']
    n_lead_days = config['n_lead_days']

    pred_path = os.path.join(BASE_DIR, f'{dataset}_{variant}', f'seed{seed}', 'results', 'predictions.npz')
    if not os.path.exists(pred_path):
        return None

    data = np.load(pred_path)
    all_results = []

    for year in test_years:
        key = f'quantiles_rmm1_{year}'
        if key not in data:
            continue

        # Get median (index 3 for 0.5 quantile)
        q1 = data[f'quantiles_rmm1_{year}']
        q2 = data[f'quantiles_rmm2_{year}']
        pred1 = q1[:, 3]  # median
        pred2 = q2[:, 3]

        gt1 = data[f'gt_rmm1_{year}']
        gt2 = data[f'gt_rmm2_{year}']
        fc1 = data[f'forecast_rmm1_{year}']
        fc2 = data[f'forecast_rmm2_{year}']

        metrics = compute_metrics_per_lead_avg(pred1, pred2, gt1, gt2, fc1, fc2, n_lead_days)
        all_results.append(metrics)

    if not all_results:
        return None

    return {
        'rmse_model': np.mean([r['rmse_model'] for r in all_results]),
        'rmse_baseline': np.mean([r['rmse_baseline'] for r in all_results]),
        'bcor_model': np.mean([r['bcor_model'] for r in all_results]),
        'bcor_baseline': np.mean([r['bcor_baseline'] for r in all_results]),
        'bmse_model': np.mean([r['bmse_model'] for r in all_results]),
        'bmse_baseline': np.mean([r['bmse_baseline'] for r in all_results]),
        'rmse_model_std': np.std([r['rmse_model'] for r in all_results]),
        'bcor_model_std': np.std([r['bcor_model'] for r in all_results]),
        'bmse_model_std': np.std([r['bmse_model'] for r in all_results]),
    }


def load_and_eval_deterministic(method, dataset, seed=42):
    """Load deterministic baseline (Kim/Silini/UAR) predictions and compute metrics."""
    config = DATASET_CONFIG[dataset]
    test_years = config['test_years']
    n_lead_days = config['n_lead_days']

    pred_path = os.path.join(BASE_DIR, 'baselines', method, dataset, f'seed{seed}', 'results', 'predictions.npz')
    if not os.path.exists(pred_path):
        return None

    data = np.load(pred_path)
    all_results = []

    for year in test_years:
        key = f'pred_rmm1_{year}'
        if key not in data:
            continue

        pred1 = data[f'pred_rmm1_{year}']
        pred2 = data[f'pred_rmm2_{year}']
        gt1 = data[f'gt_rmm1_{year}']
        gt2 = data[f'gt_rmm2_{year}']
        fc1 = data[f'fc_rmm1_{year}']
        fc2 = data[f'fc_rmm2_{year}']

        metrics = compute_metrics_per_lead_avg(pred1, pred2, gt1, gt2, fc1, fc2, n_lead_days)
        all_results.append(metrics)

    if not all_results:
        return None

    return {
        'rmse_model': np.mean([r['rmse_model'] for r in all_results]),
        'rmse_baseline': np.mean([r['rmse_baseline'] for r in all_results]),
        'bcor_model': np.mean([r['bcor_model'] for r in all_results]),
        'bcor_baseline': np.mean([r['bcor_baseline'] for r in all_results]),
        'bmse_model': np.mean([r['bmse_model'] for r in all_results]),
        'bmse_baseline': np.mean([r['bmse_baseline'] for r in all_results]),
        'rmse_model_std': np.std([r['rmse_model'] for r in all_results]),
        'bcor_model_std': np.std([r['bcor_model'] for r in all_results]),
        'bmse_model_std': np.std([r['bmse_model'] for r in all_results]),
    }


def load_and_eval_probabilistic(method, dataset):
    """Load probabilistic baseline predictions and compute metrics."""
    config = DATASET_CONFIG[dataset]
    test_years = config['test_years']
    n_lead_days = config['n_lead_days']

    # Try different naming conventions
    pred_paths = [
        os.path.join(BASE_DIR, 'baselines', 'probabilistic', dataset, 'results', f'{method}_predictions.npz'),
        os.path.join(BASE_DIR, 'baselines', 'probabilistic', dataset, 'results', f'{method}_mlp_predictions.npz'),
    ]

    pred_path = None
    for p in pred_paths:
        if os.path.exists(p):
            pred_path = p
            break

    if pred_path is None:
        return None

    data = np.load(pred_path)

    # Load forecast baseline from DBC predictions (which has it)
    dbc_path = os.path.join(BASE_DIR, f'{dataset}_norm', 'seed42', 'results', 'predictions.npz')
    dbc_data = None
    if os.path.exists(dbc_path):
        dbc_data = np.load(dbc_path)

    all_results = []

    for year in test_years:
        # Try different key formats
        pred1, pred2 = None, None

        if f'quantiles_rmm1_{year}' in data:
            q1 = data[f'quantiles_rmm1_{year}']
            q2 = data[f'quantiles_rmm2_{year}']
            # Handle different shapes: (n_forecasts, n_lead, n_quantiles) or (n_samples, n_quantiles)
            if len(q1.shape) == 3:
                # Shape: (n_forecasts, n_lead_days, n_quantiles)
                median_idx = q1.shape[2] // 2
                pred1 = q1[:, :, median_idx].flatten()
                pred2 = q2[:, :, median_idx].flatten()
            elif len(q1.shape) == 2:
                median_idx = q1.shape[1] // 2
                pred1 = q1[:, median_idx]
                pred2 = q2[:, median_idx]
            else:
                pred1, pred2 = q1.flatten(), q2.flatten()
        elif f'pred_rmm1_{year}' in data:
            pred1 = data[f'pred_rmm1_{year}'].flatten()
            pred2 = data[f'pred_rmm2_{year}'].flatten()
        elif f'mean_rmm1_{year}' in data:
            pred1 = data[f'mean_rmm1_{year}'].flatten()
            pred2 = data[f'mean_rmm2_{year}'].flatten()
        else:
            continue

        gt1 = data[f'gt_rmm1_{year}'].flatten()
        gt2 = data[f'gt_rmm2_{year}'].flatten()

        # Get forecast baseline - try multiple sources
        fc1, fc2 = None, None
        if f'fc_rmm1_{year}' in data:
            fc1 = data[f'fc_rmm1_{year}'].flatten()
            fc2 = data[f'fc_rmm2_{year}'].flatten()
        elif f'forecast_rmm1_{year}' in data:
            fc1 = data[f'forecast_rmm1_{year}'].flatten()
            fc2 = data[f'forecast_rmm2_{year}'].flatten()
        elif dbc_data is not None and f'forecast_rmm1_{year}' in dbc_data:
            fc1 = dbc_data[f'forecast_rmm1_{year}'].flatten()
            fc2 = dbc_data[f'forecast_rmm2_{year}'].flatten()
        else:
            continue

        # Make sure shapes match
        min_len = min(len(pred1), len(gt1), len(fc1))
        pred1, pred2 = pred1[:min_len], pred2[:min_len]
        gt1, gt2 = gt1[:min_len], gt2[:min_len]
        fc1, fc2 = fc1[:min_len], fc2[:min_len]

        metrics = compute_metrics_per_lead_avg(pred1, pred2, gt1, gt2, fc1, fc2, n_lead_days)
        all_results.append(metrics)

    if not all_results:
        return None

    return {
        'rmse_model': np.mean([r['rmse_model'] for r in all_results]),
        'rmse_baseline': np.mean([r['rmse_baseline'] for r in all_results]),
        'bcor_model': np.mean([r['bcor_model'] for r in all_results]),
        'bcor_baseline': np.mean([r['bcor_baseline'] for r in all_results]),
        'bmse_model': np.mean([r['bmse_model'] for r in all_results]),
        'bmse_baseline': np.mean([r['bmse_baseline'] for r in all_results]),
        'rmse_model_std': np.std([r['rmse_model'] for r in all_results]),
        'bcor_model_std': np.std([r['bcor_model'] for r in all_results]),
        'bmse_model_std': np.std([r['bmse_model'] for r in all_results]),
    }


def main():
    datasets = ['BoM', 'JMA', 'CNRM']

    # Define all methods to evaluate
    methods = {
        'RAW Forecast': 'baseline',
        'DBC (Ours)': ('dbc', 'norm'),
        'Silini': ('deterministic', 'silini'),
        'Kim': ('deterministic', 'kim'),
        'UAR': ('deterministic', 'uar'),
        'EMOS': ('probabilistic', 'emos'),
        'BMA': ('probabilistic', 'bma'),
        'BNN': ('probabilistic', 'bnn'),
        'MC Dropout': ('probabilistic', 'mcdropout'),
        'QM': ('probabilistic', 'qm_traditional'),
    }

    all_results = {}

    for dataset in datasets:
        all_results[dataset] = {}

        for method_name, method_info in methods.items():
            if method_info == 'baseline':
                # Get baseline from DBC results
                result = load_and_eval_dbc(dataset, 'norm')
                if result:
                    all_results[dataset][method_name] = {
                        'rmse': result['rmse_baseline'],
                        'bcor': result['bcor_baseline'],
                        'bmse': result['bmse_baseline'],
                    }
            elif method_info[0] == 'dbc':
                result = load_and_eval_dbc(dataset, method_info[1])
                if result:
                    all_results[dataset][method_name] = {
                        'rmse': result['rmse_model'],
                        'bcor': result['bcor_model'],
                        'bmse': result['bmse_model'],
                        'rmse_std': result['rmse_model_std'],
                        'bcor_std': result['bcor_model_std'],
                        'bmse_std': result['bmse_model_std'],
                    }
            elif method_info[0] == 'deterministic':
                result = load_and_eval_deterministic(method_info[1], dataset)
                if result:
                    all_results[dataset][method_name] = {
                        'rmse': result['rmse_model'],
                        'bcor': result['bcor_model'],
                        'bmse': result['bmse_model'],
                        'rmse_std': result['rmse_model_std'],
                        'bcor_std': result['bcor_model_std'],
                        'bmse_std': result['bmse_model_std'],
                    }
            elif method_info[0] == 'probabilistic':
                result = load_and_eval_probabilistic(method_info[1], dataset)
                if result:
                    all_results[dataset][method_name] = {
                        'rmse': result['rmse_model'],
                        'bcor': result['bcor_model'],
                        'bmse': result['bmse_model'],
                        'rmse_std': result['rmse_model_std'],
                        'bcor_std': result['bcor_model_std'],
                        'bmse_std': result['bmse_model_std'],
                    }

    # Print comparison tables
    print("=" * 120)
    print("COMPARISON TABLE - ALL METHODS (Per-Lead-Day Averaged Metrics)")
    print("=" * 120)

    # Get baseline values for computing improvements
    baselines = {}
    for dataset in datasets:
        if 'RAW Forecast' in all_results[dataset]:
            baselines[dataset] = all_results[dataset]['RAW Forecast']

    # Table 1: Raw Values
    print("\n" + "=" * 120)
    print("TABLE 1: RAW METRIC VALUES")
    print("=" * 120)

    for metric, metric_name in [('rmse', 'RMSE'), ('bcor', 'BCOR'), ('bmse', 'BMSE')]:
        print(f"\n--- {metric_name} ---")
        header = f"{'Method':<15}"
        for dataset in datasets:
            header += f" | {dataset:^12}"
        print(header)
        print("-" * 60)

        for method_name in methods.keys():
            row = f"{method_name:<15}"
            for dataset in datasets:
                if method_name in all_results[dataset] and metric in all_results[dataset][method_name]:
                    val = all_results[dataset][method_name][metric]
                    row += f" | {val:^12.4f}"
                else:
                    row += f" | {'N/A':^12}"
            print(row)

    # Table 2: Improvements over baseline
    print("\n" + "=" * 120)
    print("TABLE 2: IMPROVEMENT OVER RAW FORECAST (%)")
    print("=" * 120)

    for metric, metric_name, higher_better in [('rmse', 'RMSE', False), ('bcor', 'BCOR', True), ('bmse', 'BMSE', False)]:
        print(f"\n--- {metric_name} Improvement ---")
        header = f"{'Method':<15}"
        for dataset in datasets:
            header += f" | {dataset:^12}"
        print(header)
        print("-" * 60)

        for method_name in methods.keys():
            if method_name == 'RAW Forecast':
                continue
            row = f"{method_name:<15}"
            for dataset in datasets:
                if (method_name in all_results[dataset] and
                    metric in all_results[dataset][method_name] and
                    dataset in baselines):
                    val = all_results[dataset][method_name][metric]
                    baseline_val = baselines[dataset][metric]

                    if higher_better:
                        imp = (val - baseline_val) / abs(baseline_val) * 100
                    else:
                        imp = (baseline_val - val) / baseline_val * 100

                    row += f" | {imp:+10.2f}%"
                else:
                    row += f" | {'N/A':^12}"
            print(row)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save results
    output_path = os.path.join(BASE_DIR, 'comparison_all_methods.json')
    with open(output_path, 'w') as f:
        json.dump(convert_to_python(all_results), f, indent=2)
    print(f"\n\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
