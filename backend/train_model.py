"""Train, evaluate, and save reproducible energy-consumption models."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from backend.config import METRICS_PATH, MODEL_DIR, MODEL_PATH, get_dataset_path
from backend.data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, make_model_frame


def build_models() -> dict[str, Any]:
    def numeric_preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    FEATURE_COLUMNS,
                )
            ],
            remainder="drop",
        )

    return {
        "Linear Regression": Pipeline(
            [
                ("preprocessor", numeric_preprocessor()),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", numeric_preprocessor()),
                ("regressor", DecisionTreeRegressor(max_depth=8, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", numeric_preprocessor()),
                ("regressor", RandomForestRegressor(
                    n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
                )),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", numeric_preprocessor()),
                ("regressor", GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42
                )),
            ]
        ),
    }


def train_and_save(
    dataset_path: Path | str | None = None,
    model_path: Path | str = MODEL_PATH,
    metrics_path: Path | str = METRICS_PATH,
) -> dict[str, Any]:
    source_path = Path(dataset_path) if dataset_path else get_dataset_path()
    data = load_dataset(source_path).sort_values("Date").reset_index(drop=True)
    features = make_model_frame(data)
    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    usable = target.notna()
    features = features.loc[usable].reset_index(drop=True)
    target = target.loc[usable].reset_index(drop=True)
    if len(features) < 10:
        raise ValueError("At least 10 valid rows are required for a chronological split.")

    split_at = max(1, min(len(features) - 1, int(len(features) * 0.8)))
    x_train, x_test = features.iloc[:split_at], features.iloc[split_at:]
    y_train, y_test = target.iloc[:split_at], target.iloc[split_at:]
    scores = []
    fitted_models = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scores.append(
            {
                "model": name,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "mse": float(mean_squared_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
            }
        )
        fitted_models[name] = model

    best = min(scores, key=lambda score: score["mae"])
    best_model = fitted_models[best["model"]]
    saved_model_path = Path(model_path)
    saved_metrics_path = Path(metrics_path)
    saved_model_path.parent.mkdir(parents=True, exist_ok=True)
    saved_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best["model"],
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        saved_model_path,
    )
    results = {
        "dataset": source_path.name,
        "rows": int(len(data)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "split_method": "chronological 80/20; preprocessing fitted on training data",
        "selection_metric": "lowest test MAE",
        "best_model": best["model"],
        "best_metrics": best,
        "models": sorted(scores, key=lambda score: score["mae"]),
    }
    saved_metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved model: {saved_model_path}")
    print(f"Saved metrics: {saved_metrics_path}")
    return results


if __name__ == "__main__":
    train_and_save()
