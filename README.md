# AI-Based Smart Building Digital Twin for Energy Prediction

A beginner-friendly full-stack academic project with a Python/Flask API,
scikit-learn energy models, EDA outputs, a one-second simulated building twin,
and a React/Recharts dashboard. Live readings are **simulated**; this project
does not connect to physical IoT sensors.

## Dataset

The workspace provides `04_smart_building_energy_digital_twin2.csv` at the
project root. The requested `04_smart_building_energy_digital_twin.csv` name
was not present, so the existing file is used unchanged. To select a different
CSV, set `BUILDING_DATASET` to its absolute path before running Python.

Initial inspection: 500 rows, 14 columns, dates 2026-01-01 through 2027-05-15,
500 unique `Building_ID` values, no missing values, and no exact duplicates.
`Building_ID` is excluded from training because it is unique on every row.
Date-derived features are used instead. The small IQR outlier flags are
reported but are not clipped.

## Requirements

- Windows 10/11
- Python 3.10+ and pip
- Node.js 18+ and npm
- VS Code (recommended)

## Run in VS Code (Windows PowerShell)

Open the project root in VS Code. In the first terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m backend.eda
python -m backend.train_model
python -m unittest backend.tests.test_backend -v
python -m backend.app
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the second terminal (normally
`http://127.0.0.1:5173`). The dashboard proxies `/api` requests to Flask at
`http://127.0.0.1:5000`. Keep the backend terminal open while using the UI.
The simulator publishes a new reading every second. The dashboard fetches
updates without overlapping requests, displays the exact reading timestamp,
and shows a rolling chart of simulated live readings and ML predictions.

If PowerShell blocks venv activation, run `.\.venv\Scripts\python.exe` instead
of activating the environment, e.g.
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## Project layout

```text
backend/          Data validation, EDA, training, Flask API, simulator, tests
dataset/          Dataset-location note (original CSV remains untouched at root)
documentation/    Architecture, technical report, and presentation content
eda/              Generated plots and EDA report
frontend/         React, Vite, CSS, and Recharts dashboard
models/           Generated Joblib model and held-out metrics JSON
```

## API examples

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/building/current
Invoke-RestMethod "http://127.0.0.1:5000/api/energy/history?source=historical&limit=5"
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/energy/simulate `
  -ContentType "application/json" `
  -Body '{"Occupancy":75,"Temperature":31,"Humidity":58,"Lighting_Load":65,"HVAC_Load":52,"Computer_Load":48,"Other_Load":20}'
```

`/api/energy/history` accepts `source=historical`, `simulated`, or `predictions`.
These histories are intentionally separate. The what-if route returns actual
baseline and scenario predictions from the saved model.

## ML and validation notes

The training command fits Linear Regression, Decision Tree, Random Forest, and
Gradient Boosting models. It sorts by date, uses the earliest 80% for training
and latest 20% for testing, and chooses the lowest test MAE. Preprocessing is
inside each model pipeline to avoid fitting imputers on test rows. The
comparison includes MAE, MSE, RMSE, and R² and is written to
`models/model_metrics.json`; the selected fitted pipeline is written to
`models/energy_model.joblib`.

EDA charts and dataset summary are saved under `eda/`. See
`documentation/ARCHITECTURE.md` for data boundaries and API design,
`documentation/TECHNICAL_REPORT.md` for observed model results after training,
and `documentation/PRESENTATION.md` for slide-ready content.
