# Distributional Bias Correction for Madden-Julian Oscillation Forecast

Code and data for the paper: "Distributional Bias Correction for Madden-Julian Oscillation Forecast"

## Overview

This repository implements a Transformer-based Deep Bias Correction (DBC) model for probabilistic bias correction of Madden-Julian Oscillation (MJO) forecasts. The method provides calibrated uncertainty estimates alongside improved point predictions.

## Repository Structure

```
mjo-dbc-correction/
├── data/               # MJO forecast datasets (BOM, JMA, CNRM)
├── src/                # Main DBC model source code
├── baselines/          # Baseline methods (Kim, Silini, UAR, EMOS, BMA)
├── scripts/            # Training and evaluation scripts
├── plotting/           # Visualization scripts
└── scripts_shell/      # Shell scripts for running experiments
```

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.12
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/username/mjo-dbc-correction.git
cd mjo-dbc-correction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data

The repository includes MJO forecast data from three operational centers:

| Dataset | Lead Days | Test Years | Ensemble Members |
|---------|-----------|------------|------------------|
| BOM (Bureau of Meteorology) | 62 | 1999-2013 | 33 |
| JMA (Japan Meteorological Agency) | 33 | 1998-2012 | 5 |
| CNRM (Centre National de Recherches Meteorologiques) | 60 | 2010-2014 | 15 |

### Data Format

Each dataset directory contains:
- `forecast_rmm1_with_day0.npy`: RMM1 forecast trajectories (including day 0)
- `forecast_rmm2_with_day0.npy`: RMM2 forecast trajectories
- `ground_truth_rmm1.npy`: Observed RMM1 values
- `ground_truth_rmm2.npy`: Observed RMM2 values
- `amplitude_forecast.npy`: Forecast MJO amplitude
- `day0_info.npy`: Day 0 state [RMM1, RMM2, phase, amplitude]

## Training

### Main DBC Model

```bash
# Train on a single dataset with normalization
python scripts/train.py --dataset bom --normalize

# Train with multiple seeds for robustness
python scripts/train_multiseed.py --dataset bom --normalize --seeds 42 123 456 789 1024

# Train on all datasets
./scripts_shell/run_train_dbc.sh
```

### Baselines

```bash
# Deterministic baselines
python baselines/deterministic/kim.py --dataset BoM --seed 42
python baselines/deterministic/silini.py --dataset BoM --seed 42
python baselines/deterministic/uar.py --dataset BoM --seed 42

# Probabilistic baselines
python baselines/probabilistic/emos.py --dataset BoM
python baselines/probabilistic/bma.py --dataset BoM

# Or run all baselines
./scripts_shell/run_train_baselines.sh
```

## Evaluation

```bash
# Compute metrics for all methods
python scripts/evaluate.py

# Generate comparison tables
python scripts/generate_tables.py
```

## Generating Figures

```bash
# Metrics per lead day
python plotting/plot_metrics_per_lead.py

# Heatmap comparison
python plotting/plot_heatmap_comparison.py
```

## Model Architecture

The DBC Transformer consists of:
1. **Input Projection**: Linear layer mapping (RMM1, RMM2, Amplitude) to d_model dimensions
2. **Positional Encoding**: Learnable embeddings for lead day positions
3. **Transformer Encoder**: Multi-head self-attention with feedforward layers
4. **Non-Crossing Quantile Head**: Ensures monotonicity of quantile predictions

Key hyperparameters:
- `d_model`: 64
- `nhead`: 4
- `num_layers`: 2
- `n_quantiles`: 7 (at levels 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

## Metrics

- **BCOR**: Bivariate Correlation (higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **CRPS**: Continuous Ranked Probability Score (lower is better)
- **Coverage**: Interval coverage rate

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{author2024mjo,
  title={Probabilistic Bias Correction of MJO Forecasts using Deep Bias Correction},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MJO forecast data provided by BOM, JMA, and CNRM
