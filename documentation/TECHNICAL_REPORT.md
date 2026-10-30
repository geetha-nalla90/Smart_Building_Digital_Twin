# Technical Report — Smart Building Energy Prediction

## Summary

This project combines an exploratory analysis of the supplied historical CSV,
four held-out regression comparisons, a Flask prediction API, a synthetic live
building simulator, and an interactive React dashboard. The source CSV is
preserved unchanged and all simulated and predicted values are identified
separately from historical observations.

## Dataset and preparation

The inspected input file is `04_smart_building_energy_digital_twin2.csv`:
500 records and 14 columns, covering 2026-01-01 to 2027-05-15. It contains no
missing values or exact duplicate rows. Every row has a distinct `Building_ID`,
which is therefore excluded from model features rather than treated as a
learnable building category. `Date` is converted to calendar and cyclic
features. The numeric imputer is part of each training pipeline. The data is
sorted by date and divided into the first 400 training records and final 100
held-out records. IQR outlier flags are available in the EDA report; the
observations and target values are not clipped.

## Model results

The actual run on the 100-row chronological holdout produced:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 46.1133 | 3633.0674 | 60.2749 | 0.9725 |
| Gradient Boosting | 50.3005 | 4736.1149 | 68.8194 | 0.9641 |
| Random Forest | 55.3778 | 5312.9115 | 72.8897 | 0.9597 |
| Decision Tree | 87.1526 | 11645.3868 | 107.9138 | 0.9118 |

Linear Regression was selected because it achieved the lowest observed test
MAE. These metrics are dataset- and split-specific and are not an estimate of
performance on unseen buildings or future deployment.

## EDA outputs

`eda/eda_report.md` reports column types, missingness, duplicate rows, summary
statistics, and exploratory IQR outlier counts. Its companion charts show the
target distribution, occupancy and temperature against energy, individual
loads, numeric correlations, monthly averages, and energy by hour.

## Application behavior

The Flask API returns historical CSV rows independently from simulated
telemetry and ML prediction records. The simulator updates at one-second
intervals with gradual temperature/humidity changes and occupancy-responsive
lighting/computer loads, constrained to observed dataset ranges. The scenario
route compares a simulated current baseline against modified inputs by running
both through the saved model. Scenario changes represent model estimates, not
meter readings or guaranteed savings.

## Limitations

- One unique ID per historical row prevents building-specific model learning.
- The holdout is a single chronological split of a small dataset.
- Load and energy measurement units are not specified in the CSV; the UI
  therefore avoids assigning unsupported physical units to loads.
- Simulated readings are for demonstration only and do not control equipment.
- The what-if route compares predictions, not causal interventions.

## Reproduction

Follow the Windows setup instructions in the root `README.md`. Run
`python -m backend.eda`, `python -m backend.train_model`, and
`python -m unittest backend.tests.test_backend -v`. The generated model and
metrics are stored in `models/`.
