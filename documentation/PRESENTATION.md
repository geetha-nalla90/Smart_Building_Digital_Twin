# Presentation content

## Slide 1 — Title
**AI-Based Smart Building Digital Twin for Energy Prediction**  
An explainable full-stack demonstration combining historical data, machine
learning, simulated sensor telemetry, and what-if analysis.

## Slide 2 — Problem and objective
- Building energy demand changes with occupancy, weather, and equipment loads.
- Physical IoT devices are not available for this academic demonstration.
- Goal: explore energy drivers, train and evaluate regressors, and visualize a
  clearly labeled simulated digital twin.

## Slide 3 — Dataset
- 500 records and 14 columns, from 2026-01-01 through 2027-05-15.
- Target: `Total_Energy_Consumption`.
- The provided file is named `04_smart_building_energy_digital_twin2.csv`.
- No missing values or exact duplicates were found during initial inspection.
- `Building_ID` is unique per row and is not used as a model feature.

## Slide 4 — Data preparation
- Parse dates and derive calendar/cyclic time features.
- Use a chronological 80/20 split to keep later dates out of training.
- Imputation is included inside the estimator pipeline.
- Report potential IQR outliers without clipping target/load data.

## Slide 5 — Model comparison
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Compare held-out MAE, MSE, RMSE, and R². Results are generated from the
  actual run in `models/model_metrics.json`; do not present unrun estimates.

## Slide 6 — Digital twin and architecture
- Flask serves historical CSV records, live simulated sensor readings, and ML
  predictions as separate data sources.
- Simulator updates readings every second and keeps them within observed
  dataset ranges.
- React/Recharts dashboard polls the current twin and visualizes history.

## Slide 7 — What-if experiment
- Change occupancy, temperature, humidity, and equipment loads.
- The frontend posts the scenario to Flask.
- The saved model predicts both the live baseline and modified scenario.
- Show predicted energy change and percentage change; this is a model output,
  not a measured energy saving.

## Slide 8 — Limitations and future work
- The dataset has one unique building ID per row, so it cannot establish
  per-building learned patterns.
- The simulator is synthetic and is not an IoT or building-control system.
- More buildings, longer time series, and real sensor calibration are needed
  before operational use.
- Further work: temporal validation across seasons, uncertainty ranges, and
  authenticated deployment.
