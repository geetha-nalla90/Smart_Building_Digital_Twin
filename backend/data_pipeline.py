"""Dataset validation, feature engineering, and model input preparation."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "Total_Energy_Consumption"
RAW_NUMERIC_FEATURES = [
    "Hour",
    "Temperature",
    "Humidity",
    "Occupancy",
    "Lighting_Load",
    "HVAC_Load",
    "Computer_Load",
    "Other_Load",
    "Previous_Energy_Consumption",
    "Weekend",
    "Holiday",
]
FEATURE_COLUMNS = RAW_NUMERIC_FEATURES + [
    "Month",
    "DayOfWeek",
    "DayOfYear",
    "Hour_Sin",
    "Hour_Cos",
    "Month_Sin",
    "Month_Cos",
]
REQUIRED_COLUMNS = [
    "Building_ID",
    "Date",
    *RAW_NUMERIC_FEATURES,
    TARGET_COLUMN,
]


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load and validate the source CSV without modifying it."""
    data = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if data["Date"].isna().any():
        raise ValueError("Date contains values that cannot be parsed.")
    if data[TARGET_COLUMN].isna().all():
        raise ValueError(f"{TARGET_COLUMN} contains no usable values.")
    return data


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convert Date into stable numeric features and drop the high-cardinality ID."""
    featured = data.copy()
    dates = pd.to_datetime(featured["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Date contains values that cannot be parsed.")
    featured["Month"] = dates.dt.month
    featured["DayOfWeek"] = dates.dt.dayofweek
    featured["DayOfYear"] = dates.dt.dayofyear
    featured["Hour_Sin"] = np.sin(2 * np.pi * featured["Hour"] / 24)
    featured["Hour_Cos"] = np.cos(2 * np.pi * featured["Hour"] / 24)
    featured["Month_Sin"] = np.sin(2 * np.pi * featured["Month"] / 12)
    featured["Month_Cos"] = np.cos(2 * np.pi * featured["Month"] / 12)
    return featured


def make_model_frame(data: pd.DataFrame) -> pd.DataFrame:
    featured = add_time_features(data)
    return featured.reindex(columns=FEATURE_COLUMNS)


def validate_reading(reading: dict[str, Any]) -> dict[str, Any]:
    """Validate API-controlled sensor values against observed domain ranges."""
    bounds = {
        "Hour": (0, 23),
        "Temperature": (18, 42),
        "Humidity": (25, 95),
        "Occupancy": (0, 100),
        "Lighting_Load": (0, 100),
        "HVAC_Load": (0, 100),
        "Computer_Load": (0, 100),
        "Other_Load": (0, 100),
        "Previous_Energy_Consumption": (0, 5000),
        "Weekend": (0, 1),
        "Holiday": (0, 1),
    }
    clean: dict[str, Any] = {}
    for field, (minimum, maximum) in bounds.items():
        if field not in reading:
            continue
        value = reading[field]
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a number.")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be a number.") from error
        if not np.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"{field} must be between {minimum} and {maximum}.")
        clean[field] = number
    if "Date" in reading:
        parsed_date = pd.to_datetime(reading["Date"], errors="coerce")
        if pd.isna(parsed_date):
            raise ValueError("Date must be a valid date.")
        clean["Date"] = parsed_date.strftime("%Y-%m-%d")
    if "Building_ID" in reading:
        building_id = reading["Building_ID"]
        if not isinstance(building_id, str) or not building_id.strip():
            raise ValueError("Building_ID must be a non-empty string.")
        clean["Building_ID"] = building_id.strip()
    return clean


def input_frame(reading: dict[str, Any]) -> pd.DataFrame:
    normalized = validate_reading(reading)
    defaults: dict[str, Any] = {
        "Building_ID": "B001",
        "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "Weekend": 0,
        "Holiday": 0,
    }
    defaults.update({field: 0.0 for field in RAW_NUMERIC_FEATURES})
    defaults.update(normalized)
    if "Hour" not in normalized:
        defaults["Hour"] = pd.Timestamp.now().hour
    row = pd.DataFrame([defaults])
    return make_model_frame(row)
