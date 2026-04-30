#!/usr/bin/env python3
"""
Generate unified result tables for MJO Probabilistic Correction paper.
Reads metrics from all model outputs and creates Markdown/LaTeX tables.
"""

import json
import os
from pathlib import Path

# Auto-detect repository root from file location
BASE_DIR = Path(__file__).resolve().parent.parent

# Baseline values (from raw ensemble forecasts, per-year averaged)
BASELINES = {
    "BoM": {"bcor": 0.4589, "rmse": 1.2233},
    "JMA": {"bcor": 0.6336, "rmse": 1.0184},
    "CNRM": {"bcor": 0.4431, "rmse": 1.3062},
}

def load_json(path):
    """Load JSON file safely."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

def compute_improvement(model_val, baseline_val, metric):
    """Compute improvement percentage."""
    if metric == "bcor":
        # Higher is better
        return (model_val - baseline_val) / baseline_val * 100
    else:
        # Lower is better (RMSE)
        return (baseline_val - model_val) / baseline_val * 100

def collect_dbc_results():
    """Collect DBC model results from metrics_recomputed.json files."""
    results = []

    for dataset in ["BoM", "JMA", "CNRM"]:
        for norm in ["norm", "no_norm"]:
            path = BASE_DIR / f"{dataset}_{norm}" / "results" / "metrics_recomputed.json"
            data = load_json(path)
            if data:
                method = f"DBC-T ({norm.replace('_', ' ')})"
                results.append({
                    "method": method,
                    "dataset": dataset,
                    "bcor": data.get("bcor_model_mean", 0),
                    "rmse": data.get("rmse_model_mean", 0),
                    "bcor_imp": data.get("bcor_improvement_mean", 0),
                    "rmse_imp": data.get("rmse_improvement_mean", 0),
                    "bcor_std": data.get("bcor_model_std", 0),
                    "rmse_std": data.get("rmse_model_std", 0),
                })
    return results

def collect_deterministic_baselines():
    """Collect deterministic baseline results."""
    results = []
    methods = {
        "kim": "Kim et al.",
        "silini": "Silini et al.",
        "uar": "UAR",
    }

    for method_key, method_name in methods.items():
        for dataset in ["BoM", "JMA", "CNRM"]:
            path = BASE_DIR / "baselines" / method_key / dataset / "results" / "summary.json"
            data = load_json(path)
            if data:
                baseline = BASELINES[dataset]
                # Compute model values from improvement
                bcor_imp = data.get("bcor_improvement_mean", 0)
                rmse_imp = data.get("rmse_improvement_mean", 0)
                # Reverse compute: bcor_model = baseline * (1 + imp/100)
                bcor_model = baseline["bcor"] * (1 + bcor_imp / 100)
                rmse_model = baseline["rmse"] * (1 - rmse_imp / 100)

                results.append({
                    "method": method_name,
                    "dataset": dataset,
                    "bcor": bcor_model,
                    "rmse": rmse_model,
                    "bcor_imp": bcor_imp,
                    "rmse_imp": rmse_imp,
                    "bcor_std": data.get("bcor_improvement_std", 0),
                    "rmse_std": data.get("rmse_improvement_std", 0),
                })
    return results

def collect_probabilistic_baselines():
    """Collect probabilistic baseline results."""
    results = []
    methods = {
        "emos_summary.json": "EMOS",
        "bma_summary.json": "BMA",
        "bnn_mlp_summary.json": "BNN-MLP",
        "gp_summary.json": "GP",
        "mcdropout_summary.json": "MC Dropout",
    }

    for dataset in ["BoM", "JMA", "CNRM"]:
        prob_dir = BASE_DIR / "baselines" / "probabilistic" / dataset / "results"
        if not prob_dir.exists():
            continue

        for filename, method_name in methods.items():
            path = prob_dir / filename
            data = load_json(path)
            if data and "mean_bcor" in data:
                baseline = BASELINES[dataset]
                bcor_model = data.get("mean_bcor", 0)
                rmse_model = data.get("mean_rmse", 0)
                crps = data.get("mean_crps", 0)

                bcor_imp = compute_improvement(bcor_model, baseline["bcor"], "bcor")
                rmse_imp = compute_improvement(rmse_model, baseline["rmse"], "rmse")

                results.append({
                    "method": method_name,
                    "dataset": dataset,
                    "bcor": bcor_model,
                    "rmse": rmse_model,
                    "crps": crps,
                    "bcor_imp": bcor_imp,
                    "rmse_imp": rmse_imp,
                    "bcor_std": data.get("std_bcor", 0),
                    "rmse_std": data.get("std_rmse", 0),
                })
    return results

def generate_markdown_table(results, title):
    """Generate a Markdown table from results."""
    lines = [f"## {title}\n"]

    # Group by method
    methods = sorted(set(r["method"] for r in results))
    datasets = ["BoM", "JMA", "CNRM"]

    # Header
    lines.append("| Method | " + " | ".join([f"{d} BCOR | {d} RMSE" for d in datasets]) + " |")
    lines.append("|--------|" + "|".join(["--------|--------" for _ in datasets]) + "|")

    for method in methods:
        row = [method]
        for dataset in datasets:
            matching = [r for r in results if r["method"] == method and r["dataset"] == dataset]
            if matching:
                r = matching[0]
                bcor_str = f"{r['bcor']:.3f}"
                rmse_str = f"{r['rmse']:.3f}"
                row.append(bcor_str)
                row.append(rmse_str)
            else:
                row.extend(["-", "-"])
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

def generate_improvement_table(results, title):
    """Generate improvement percentage table."""
    lines = [f"## {title}\n"]

    methods = sorted(set(r["method"] for r in results))
    datasets = ["BoM", "JMA", "CNRM"]

    # Header
    lines.append("| Method | " + " | ".join([f"{d} BCOR↑ | {d} RMSE↓" for d in datasets]) + " |")
    lines.append("|--------|" + "|".join(["--------|--------" for _ in datasets]) + "|")

    for method in methods:
        row = [method]
        for dataset in datasets:
            matching = [r for r in results if r["method"] == method and r["dataset"] == dataset]
            if matching:
                r = matching[0]
                bcor_str = f"+{r['bcor_imp']:.1f}%" if r['bcor_imp'] > 0 else f"{r['bcor_imp']:.1f}%"
                rmse_str = f"+{r['rmse_imp']:.1f}%" if r['rmse_imp'] > 0 else f"{r['rmse_imp']:.1f}%"
                row.append(bcor_str)
                row.append(rmse_str)
            else:
                row.extend(["-", "-"])
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

def generate_comprehensive_table(all_results):
    """Generate a comprehensive table with all methods."""
    lines = ["## Comprehensive Results Table\n"]
    lines.append("All values are per-year averaged. BCOR: higher is better. RMSE: lower is better.\n")

    # Sort results by method type, then method name
    method_order = [
        "DBC-T (norm)", "DBC-T (no norm)",  # Our methods
        "Kim et al.", "Silini et al.", "UAR",  # Deterministic
        "EMOS", "BMA", "BNN-MLP", "GP", "MC Dropout",  # Probabilistic
    ]

    datasets = ["BoM", "JMA", "CNRM"]

    # Header
    lines.append("| Method | Type | " + " | ".join([f"{d} BCOR | {d} RMSE | {d} BCOR% | {d} RMSE%" for d in datasets]) + " |")
    lines.append("|--------|------|" + "|".join(["-------|-------|--------|--------" for _ in datasets]) + "|")

    # Add baseline row
    row = ["Raw Ensemble", "Baseline"]
    for d in datasets:
        b = BASELINES[d]
        row.extend([f"{b['bcor']:.3f}", f"{b['rmse']:.3f}", "-", "-"])
    lines.append("| " + " | ".join(row) + " |")

    for method in method_order:
        matching = [r for r in all_results if r["method"] == method]
        if not matching:
            continue

        # Determine type
        if "DBC" in method:
            mtype = "Proposed"
        elif method in ["Kim et al.", "Silini et al.", "UAR"]:
            mtype = "Determin."
        else:
            mtype = "Probab."

        row = [method, mtype]
        for dataset in datasets:
            dm = [r for r in matching if r["dataset"] == dataset]
            if dm:
                r = dm[0]
                bcor_str = f"{r['bcor']:.3f}"
                rmse_str = f"{r['rmse']:.3f}"
                bcor_imp_str = f"+{r['bcor_imp']:.1f}" if r['bcor_imp'] > 0 else f"{r['bcor_imp']:.1f}"
                rmse_imp_str = f"+{r['rmse_imp']:.1f}" if r['rmse_imp'] > 0 else f"{r['rmse_imp']:.1f}"
                row.extend([bcor_str, rmse_str, bcor_imp_str, rmse_imp_str])
            else:
                row.extend(["-", "-", "-", "-"])
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

def generate_latex_table(all_results):
    """Generate LaTeX table."""
    lines = [
        "% LaTeX Table for MJO Probabilistic Correction Results",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Performance comparison across all methods and datasets. BCOR\\% and RMSE\\% show improvement over raw ensemble.}",
        "\\label{tab:results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccccccccc}",
        "\\toprule",
        "& & \\multicolumn{4}{c}{BoM} & \\multicolumn{4}{c}{JMA} & \\multicolumn{4}{c}{CNRM} \\\\",
        "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10} \\cmidrule(lr){11-14}",
        "Method & Type & BCOR & RMSE & $\\Delta$B & $\\Delta$R & BCOR & RMSE & $\\Delta$B & $\\Delta$R & BCOR & RMSE & $\\Delta$B & $\\Delta$R \\\\",
        "\\midrule",
    ]

    method_order = [
        "DBC-T (norm)", "DBC-T (no norm)",
        "Kim et al.", "Silini et al.", "UAR",
        "EMOS", "BMA", "BNN-MLP", "GP",
    ]

    datasets = ["BoM", "JMA", "CNRM"]

    # Add baseline row
    row_parts = ["Raw Ensemble", "Baseline"]
    for d in datasets:
        b = BASELINES[d]
        row_parts.extend([f"{b['bcor']:.3f}", f"{b['rmse']:.3f}", "--", "--"])
    lines.append(" & ".join(row_parts) + " \\\\")
    lines.append("\\midrule")

    for method in method_order:
        matching = [r for r in all_results if r["method"] == method]
        if not matching:
            continue

        if "DBC" in method:
            mtype = "Proposed"
        elif method in ["Kim et al.", "Silini et al.", "UAR"]:
            mtype = "Det."
        else:
            mtype = "Prob."

        row_parts = [method.replace("_", "\\_"), mtype]
        for dataset in datasets:
            dm = [r for r in matching if r["dataset"] == dataset]
            if dm:
                r = dm[0]
                bcor_str = f"{r['bcor']:.3f}"
                rmse_str = f"{r['rmse']:.3f}"
                bcor_imp = r['bcor_imp']
                rmse_imp = r['rmse_imp']
                bcor_imp_str = f"+{bcor_imp:.1f}" if bcor_imp > 0 else f"{bcor_imp:.1f}"
                rmse_imp_str = f"+{rmse_imp:.1f}" if rmse_imp > 0 else f"{rmse_imp:.1f}"
                row_parts.extend([bcor_str, rmse_str, bcor_imp_str, rmse_imp_str])
            else:
                row_parts.extend(["--", "--", "--", "--"])
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])

    return "\n".join(lines)

def main():
    print("Collecting results...")

    # Collect all results
    dbc_results = collect_dbc_results()
    det_results = collect_deterministic_baselines()
    prob_results = collect_probabilistic_baselines()

    all_results = dbc_results + det_results + prob_results

    print(f"\nCollected {len(dbc_results)} DBC results")
    print(f"Collected {len(det_results)} deterministic baseline results")
    print(f"Collected {len(prob_results)} probabilistic baseline results")
    print(f"Total: {len(all_results)} results")

    # Generate tables
    print("\n" + "="*80)
    print(generate_comprehensive_table(all_results))
    print("\n" + "="*80)
    print(generate_improvement_table(dbc_results + det_results, "Improvement Over Baseline (Deterministic Methods)"))
    print("\n" + "="*80)
    print(generate_improvement_table(prob_results, "Improvement Over Baseline (Probabilistic Methods)"))
    print("\n" + "="*80 + "\n")
    print(generate_latex_table(all_results))

    # Save to file
    output_path = BASE_DIR / "results_tables.md"
    with open(output_path, "w") as f:
        f.write("# MJO Probabilistic Correction - Results Tables\n\n")
        f.write(f"Generated automatically from model outputs.\n\n")
        f.write("## Baseline Values (Raw Ensemble)\n\n")
        f.write("| Dataset | BCOR | RMSE |\n")
        f.write("|---------|------|------|\n")
        for d, v in BASELINES.items():
            f.write(f"| {d} | {v['bcor']:.3f} | {v['rmse']:.3f} |\n")
        f.write("\n")
        f.write(generate_comprehensive_table(all_results))
        f.write("\n\n")
        f.write(generate_improvement_table(dbc_results + det_results, "Improvement Over Baseline (Deterministic Methods)"))
        f.write("\n\n")
        f.write(generate_improvement_table(prob_results, "Improvement Over Baseline (Probabilistic Methods)"))
        f.write("\n\n## LaTeX Table\n\n```latex\n")
        f.write(generate_latex_table(all_results))
        f.write("\n```\n")

    print(f"\nTables saved to: {output_path}")

if __name__ == "__main__":
    main()
