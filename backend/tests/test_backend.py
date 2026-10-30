import unittest
from pathlib import Path
import time

import pandas as pd

from backend.config import get_dataset_path, METRICS_PATH, MODEL_PATH
from backend.data_pipeline import FEATURE_COLUMNS, input_frame, load_dataset
from backend.simulator import BuildingSimulator
from backend.train_model import train_and_save


ROOT = Path(__file__).resolve().parents[2]


class DatasetAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_path = get_dataset_path()
        cls.data = load_dataset(cls.dataset_path)
        cls.metrics = train_and_save(cls.dataset_path, MODEL_PATH, METRICS_PATH)

    def test_dataset_has_target_and_no_missing_values(self):
        self.assertIn("Total_Energy_Consumption", self.data.columns)
        self.assertEqual(int(self.data.isna().sum().sum()), 0)
        self.assertEqual(int(self.data.duplicated().sum()), 0)

    def test_preprocessing_builds_expected_finite_features(self):
        frame = input_frame(
            {
                "Date": "2026-10-30",
                "Hour": 10,
                "Temperature": 27,
                "Humidity": 60,
                "Occupancy": 40,
                "Lighting_Load": 50,
                "HVAC_Load": 35,
                "Computer_Load": 30,
                "Other_Load": 15,
                "Previous_Energy_Consumption": 600,
                "Weekend": 0,
                "Holiday": 0,
            }
        )
        self.assertEqual(list(frame.columns), FEATURE_COLUMNS)
        self.assertTrue(frame.notna().all().all())

    def test_training_saves_comparable_models_and_prediction(self):
        self.assertTrue(MODEL_PATH.is_file())
        self.assertEqual(self.metrics["rows"], len(self.data))
        self.assertEqual(self.metrics["train_rows"] + self.metrics["test_rows"], len(self.data))
        self.assertEqual(len(self.metrics["models"]), 4)
        self.assertEqual(
            self.metrics["best_model"],
            min(self.metrics["models"], key=lambda item: item["mae"])["model"],
        )

    def test_invalid_sensor_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Occupancy"):
            input_frame({"Occupancy": 101})
        with self.assertRaisesRegex(ValueError, "Temperature"):
            input_frame({"Temperature": float("nan")})


class SimulatorTests(unittest.TestCase):
    def test_simulator_produces_bounded_simulated_readings(self):
        dataset = load_dataset(get_dataset_path())
        simulator = BuildingSimulator(dataset)
        reading = simulator.current()
        self.assertEqual(reading["source"], "simulated")
        for column in (
            "Temperature", "Humidity", "Occupancy", "Lighting_Load",
            "HVAC_Load", "Computer_Load", "Other_Load",
        ):
            self.assertGreaterEqual(reading[column], dataset[column].min())
            self.assertLessEqual(reading[column], dataset[column].max())
        self.assertEqual(len(simulator.history(10)), 1)

    def test_simulator_generates_readings_on_its_interval(self):
        dataset = load_dataset(get_dataset_path())
        simulator = BuildingSimulator(dataset, interval_seconds=0.02)
        simulator.start()
        time.sleep(0.08)
        simulator.stop()
        self.assertGreaterEqual(len(simulator.history(10)), 2)
        readings = simulator.history(10)
        self.assertNotEqual(readings[-1]["timestamp"], readings[-2]["timestamp"])
        for column in (
            "Occupancy", "Temperature", "Humidity", "Lighting_Load",
            "HVAC_Load", "Computer_Load", "Other_Load",
        ):
            self.assertLess(
                abs(readings[-1][column] - readings[-2][column]), 5,
                f"{column} changed too sharply between simulator ticks",
            )


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.app import create_app

        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.simulator = cls.app.extensions["building_simulator"]
        cls.simulator.start()

    @classmethod
    def tearDownClass(cls):
        cls.simulator.stop()

    def test_current_history_and_metrics_endpoints(self):
        current = self.client.get("/api/building/current")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json["data_source"], "simulated")
        self.assertEqual(current.json["data_label"], "SIMULATED REAL-TIME DATA")
        self.assertGreater(current.json["predicted_energy_consumption"], 0)
        self.assertEqual(
            current.json["last_updated"], current.json["current"]["timestamp"]
        )

        history = self.client.get("/api/energy/history?source=historical&limit=5")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json["data_source"], "historical_csv")
        self.assertEqual(len(history.json["rows"]), 5)

        metrics = self.client.get("/api/model/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(len(metrics.json["models"]), 4)

    def test_current_endpoint_tracks_one_second_simulator_ticks(self):
        first = self.client.get("/api/building/current").json
        time.sleep(1.1)
        second = self.client.get("/api/building/current").json
        self.assertNotEqual(first["last_updated"], second["last_updated"])
        self.assertEqual(second["last_updated"], second["current"]["timestamp"])
        self.assertIsInstance(second["predicted_energy_consumption"], float)
        predictions = self.client.get(
            "/api/energy/history?source=predictions&limit=10"
        ).json["rows"]
        matching = [
            row for row in predictions if row["timestamp"] == second["last_updated"]
        ]
        self.assertTrue(matching)
        self.assertEqual(
            matching[-1]["predicted_energy"],
            second["predicted_energy_consumption"],
        )

    def test_prediction_and_what_if_simulation_use_model(self):
        payload = {"Occupancy": 80, "Temperature": 32, "Humidity": 55}
        prediction = self.client.post("/api/energy/predict", json=payload)
        self.assertEqual(prediction.status_code, 200)
        self.assertGreater(prediction.json["predicted_energy_consumption"], 0)

        simulation = self.client.post("/api/energy/simulate", json=payload)
        self.assertEqual(simulation.status_code, 200)
        self.assertAlmostEqual(
            simulation.json["energy_difference"],
            simulation.json["scenario_prediction"] - simulation.json["baseline_prediction"],
        )

        history = self.client.get("/api/energy/history?source=predictions")
        self.assertGreaterEqual(len(history.json["rows"]), 4)

    def test_invalid_requests_return_actionable_errors(self):
        invalid_value = self.client.post("/api/energy/predict", json={"Occupancy": 101})
        self.assertEqual(invalid_value.status_code, 400)
        self.assertIn("Occupancy", invalid_value.json["error"])

        invalid_field = self.client.post("/api/energy/simulate", json={"unexpected": 3})
        self.assertEqual(invalid_field.status_code, 400)
        invalid_source = self.client.get("/api/energy/history?source=unknown")
        self.assertEqual(invalid_source.status_code, 400)
        malformed = self.client.post(
            "/api/energy/predict", data="[broken", content_type="application/json"
        )
        self.assertEqual(malformed.status_code, 400)


if __name__ == "__main__":
    unittest.main()
