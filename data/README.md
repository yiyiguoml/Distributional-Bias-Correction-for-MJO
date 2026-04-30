# MJO Forecast Data

This directory contains MJO forecast data from three operational centers.

## Data Format

All data files are NumPy arrays (.npy format).

### Forecast Files
- `forecast_rmm1_with_day0.npy`: Shape (n_samples, n_lead_days+1)
  - RMM1 forecast values including day 0 (analysis)
- `forecast_rmm2_with_day0.npy`: Shape (n_samples, n_lead_days+1)
  - RMM2 forecast values including day 0 (analysis)

### Ground Truth
- `ground_truth_rmm1.npy`: Shape (n_samples, n_lead_days)
  - Observed RMM1 values for each lead day
- `ground_truth_rmm2.npy`: Shape (n_samples, n_lead_days)
  - Observed RMM2 values for each lead day

### Additional Data
- `amplitude_forecast.npy`: Shape (n_samples, n_lead_days)
  - Forecast MJO amplitude
- `day0_info.npy`: Shape (n_samples, 4)
  - Day 0 state: [RMM1, RMM2, phase, amplitude]
- `times.npy`: Shape (n_samples,)
  - Datetime array for each forecast initialization

### Ensemble Data (for BMA baseline)
- `rmm1_ensemble.npy` / `ensemble_rmm1.npy`: Shape (n_samples, n_lead_days, n_members)
- `rmm2_ensemble.npy` / `ensemble_rmm2.npy`: Shape (n_samples, n_lead_days, n_members)

## Dataset Statistics

| Dataset | Samples | Lead Days | Test Years | Ensemble Members |
|---------|---------|-----------|------------|------------------|
| BOM | ~12,000 | 62 | 1999-2013 (15 years) | 33 |
| JMA | ~11,000 | 33 | 1998-2012 (15 years) | 5 |
| CNRM | ~8,000 | 60 | 2010-2014 (5 years) | 15 |

## Data Sources

- **BOM**: Bureau of Meteorology, Australia
- **JMA**: Japan Meteorological Agency
- **CNRM**: Centre National de Recherches Meteorologiques, France
