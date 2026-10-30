# Smart Building Digital Twin — Architecture

## Components and data boundaries

```text
Root CSV (historical observations)
       | load / EDA / chronological training
       +--> backend/data_pipeline.py --> backend/train_model.py
       |                                    | saved model + metrics
       |                                    v
       +--> backend/app.py <---------- saved Joblib model
                 |             |
                 |             +--> simulated live readings (in-memory, 1 s)
                 |             +--> ML prediction history (in-memory, separate)
                 v
            JSON REST API <--> React + Recharts dashboard
```

The historical CSV remains read-only. Simulator readings are generated and
bounded by the observed sensor ranges, updated with damped gradual changes,
and tagged `SIMULATED REAL-TIME DATA`. The simulator invokes the saved
scikit-learn model on each new tick and stores the prediction beside its
matching timestamp. The API exposes each source independently. The React
dashboard polls the current endpoint every second after the prior request
completes, preventing overlapping fetches.

## Model path

`Date` is parsed into calendar and cyclic numeric features. The unique
`Building_ID` is retained for display but excluded from training because every
row has a different ID. Numeric imputation is part of each model pipeline and
is fit only on the training partition. Rows are sorted by date and divided into
an 80/20 chronological train/test split. Models are compared by MAE; MAE, MSE,
RMSE, and R² are saved for every candidate. IQR flags are reported in EDA but
not clipped, preserving observations and target values.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/building/current` | Latest simulated reading and its ML prediction |
| POST | `/api/energy/predict` | ML prediction for supplied sensor values |
| GET | `/api/energy/history?source=historical` | Historical CSV observations |
| GET | `/api/energy/history?source=simulated` | Recent generated readings |
| GET | `/api/energy/history?source=predictions` | Recent model prediction records |
| POST | `/api/energy/simulate` | Baseline and scenario predictions using the model |
| GET | `/api/model/metrics` | Test-set model comparison |

Simulation is stateless with respect to the CSV and predictions; the live
simulator and prediction histories are held separately in process memory.
