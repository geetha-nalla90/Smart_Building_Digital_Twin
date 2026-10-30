"""Flask API for historical data, simulated telemetry, and ML predictions."""

from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
import threading
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from backend.config import METRICS_PATH, MODEL_PATH, get_dataset_path
from backend.data_pipeline import input_frame, load_dataset, validate_reading
from backend.simulator import BuildingSimulator


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
PREDICTION_HISTORY_LIMIT = 500
ALLOWED_INPUTS = {
    "Building_ID", "Date", "Hour", "Temperature", "Humidity", "Occupancy",
    "Lighting_Load", "HVAC_Load", "Computer_Load", "Other_Load",
    "Previous_Energy_Consumption", "Weekend", "Holiday",
}


def create_app(
    model_path: Path | str = MODEL_PATH,
    dataset_path: Path | str | None = None,
    metrics_path: Path | str = METRICS_PATH,
) -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    data_path = Path(dataset_path) if dataset_path else get_dataset_path()
    historical_data = load_dataset(data_path)
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_file}. Run `python -m backend.train_model` first."
        )
    model_bundle = joblib.load(model_file)
    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        raise ValueError(f"Model artifact has an invalid format: {model_file}")
    model = model_bundle["model"]
    simulator = BuildingSimulator(historical_data)
    prediction_history: deque[dict[str, Any]] = deque(maxlen=PREDICTION_HISTORY_LIMIT)
    live_predictions: dict[str, float] = {}
    prediction_lock = threading.Lock()

    def predict_reading(reading: dict[str, Any], source: str) -> float:
        normalized = validate_reading(reading)
        print("Starting model prediction...", flush=True)
        prediction = float(model.predict(input_frame(normalized))[0])
        print("Model prediction completed:", prediction, flush=True)
        prediction_history.append(
            {
                "timestamp": reading.get(
                    "timestamp",
                    datetime.now().astimezone().isoformat(timespec="milliseconds"),
                ),
                "predicted_energy": prediction,
                "source": source,
                "input": normalized,
            }
        )
        print("PREDICT_READING: Returning prediction", flush=True)
        return prediction

    def live_prediction_for(reading: dict[str, Any]) -> float:
        print("LIVE PREDICTION: Entered function", flush=True)
        timestamp = reading["timestamp"]
        print("LIVE PREDICTION: Waiting for lock", flush=True)
        with prediction_lock:
            print("LIVE PREDICTION: Lock acquired", flush=True)
            if timestamp not in live_predictions:
                live_predictions[timestamp] = predict_reading(
                    reading, "simulated_live_reading"
                )
            while len(live_predictions) > 10:
                oldest_timestamp = next(iter(live_predictions))
                del live_predictions[oldest_timestamp]
            print("LIVE PREDICTION: Returning cached prediction", flush=True)
            return live_predictions[timestamp]

    def update_live_prediction(reading: dict[str, Any]) -> None:
        live_prediction_for(reading)

    simulator.on_update = update_live_prediction

    def current_prediction() -> tuple[dict[str, Any], float]:
        print("STEP 1: Getting current reading...", flush=True)

        reading = simulator.current()

        print("STEP 2: Current reading received.", flush=True)

        prediction = live_prediction_for(reading)

        print("STEP 3: Prediction received.", flush=True)

        return reading, prediction

    @app.after_request
    def add_api_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.errorhandler(400)
    def bad_request(error):
        message = getattr(error, "description", "Invalid request.")
        return jsonify({"error": str(message)}), 400

    @app.errorhandler(ValueError)
    def invalid_input(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(500)
    def internal_error(error):
        LOGGER.exception("Unhandled API error: %s", error)
        return jsonify({"error": "Internal server error. Check backend logs."}), 500

    @app.get("/api/building/current")
    def get_current_building():
        print("TEST ROUTE ENTERED", flush=True)

        return jsonify({
            "status": "success",
            "message": "Current API is responding"
        })

    @app.post("/api/energy/predict")
    def predict_energy():
        payload = request.get_json()
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        unknown = sorted(set(payload) - ALLOWED_INPUTS)
        if unknown:
            return jsonify({"error": f"Unknown input fields: {unknown}"}), 400
        reading = simulator.current()
        reading.update(payload)
        prediction = predict_reading(reading, "prediction_request")
        return jsonify(
            {
                "predicted_energy_consumption": prediction,
                "model": model_bundle.get("model_name", "Saved model"),
                "data_source": "ml_prediction",
            }
        )

    @app.get("/api/energy/history")
    def get_energy_history():
        source = request.args.get("source", "historical")
        if source not in {"historical", "simulated", "predictions"}:
            return jsonify(
                {"error": "source must be historical, simulated, or predictions."}
            ), 400
        try:
            limit = int(request.args.get("limit", 120))
        except ValueError:
            return jsonify({"error": "limit must be an integer."}), 400
        if not 1 <= limit <= 500:
            return jsonify({"error": "limit must be between 1 and 500."}), 400

        if source == "historical":
            rows = historical_data.sort_values("Date").tail(limit).copy()
            rows["Date"] = pd.to_datetime(rows["Date"]).dt.strftime("%Y-%m-%d")
            return jsonify(
                {
                    "source": "historical",
                    "data_source": "historical_csv",
                    "rows": rows.to_dict(orient="records"),
                }
            )
        if source == "simulated":
            return jsonify(
                {
                    "source": "simulated",
                    "data_source": "simulated_live_readings",
                    "rows": simulator.history(limit),
                }
            )
        return jsonify(
            {
                "source": "predictions",
                "data_source": "ml_predictions",
                "rows": list(prediction_history)[-limit:],
            }
        )

    @app.post("/api/energy/simulate")
    def simulate_energy():
        payload = request.get_json()
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        unknown = sorted(set(payload) - ALLOWED_INPUTS)
        if unknown:
            return jsonify({"error": f"Unknown input fields: {unknown}"}), 400
        baseline_reading = simulator.current()
        scenario_reading = {**baseline_reading, **payload}
        baseline = predict_reading(baseline_reading, "what_if_baseline")
        scenario = predict_reading(scenario_reading, "what_if_scenario")
        difference = scenario - baseline
        percentage = (difference / abs(baseline) * 100) if baseline else 0.0
        return jsonify(
            {
                "baseline_prediction": baseline,
                "scenario_prediction": scenario,
                "energy_difference": difference,
                "percentage_change": percentage,
                "model": model_bundle.get("model_name", "Saved model"),
                "data_source": "ml_prediction",
            }
        )

    @app.get("/api/model/metrics")
    def get_model_metrics():
        path = Path(metrics_path)
        if not path.is_file():
            return jsonify({"error": f"Model metrics not found at {path}."}), 404
        return jsonify(json.loads(path.read_text(encoding="utf-8")))

    app.extensions["building_simulator"] = simulator
    return app


app = create_app()


app.extensions["building_simulator"].start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
