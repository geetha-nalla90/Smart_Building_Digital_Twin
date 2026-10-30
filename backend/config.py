from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
EDA_DIR = ROOT_DIR / "eda"
OUTPUT_DIR = ROOT_DIR / "documentation"
DATASET_CANDIDATES = (
    ROOT_DIR / "04_smart_building_energy_digital_twin.csv",
    ROOT_DIR / "04_smart_building_energy_digital_twin2.csv",
)


def get_dataset_path() -> Path:
    configured_path = os.environ.get("BUILDING_DATASET")
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"BUILDING_DATASET does not exist: {path}")
        return path
    for path in DATASET_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Dataset not found. Set BUILDING_DATASET or place the CSV in the project root."
    )


MODEL_PATH = MODEL_DIR / "energy_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
