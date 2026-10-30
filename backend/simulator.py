"""Deterministic, dataset-bounded simulation of live building telemetry."""

from collections import deque
from datetime import datetime
import threading
import time
from typing import Any, Callable
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
class BuildingSimulator:
    def __init__(
        self,
        dataset: pd.DataFrame,
        interval_seconds: float = 1,
        history_size: int = 720,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.interval_seconds = interval_seconds
        self.on_update = on_update
        self._data = dataset
        self._rng = np.random.default_rng(42)
        self._lock = threading.Lock()
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._last_step = 0.0
        self._limits = {
            column: (float(dataset[column].min()), float(dataset[column].max()))
            for column in (
                "Temperature", "Humidity", "Occupancy", "Lighting_Load",
                "HVAC_Load", "Computer_Load", "Other_Load",
                "Previous_Energy_Consumption",
            )
        }
        self._state: dict[str, Any] = {
            "Building_ID": str(dataset["Building_ID"].iloc[0]),
            "Temperature": float(dataset["Temperature"].median()),
            "Humidity": float(dataset["Humidity"].median()),
            "Occupancy": float(dataset["Occupancy"].median()),
            "Lighting_Load": float(dataset["Lighting_Load"].median()),
            "HVAC_Load": float(dataset["HVAC_Load"].median()),
            "Computer_Load": float(dataset["Computer_Load"].median()),
            "Other_Load": float(dataset["Other_Load"].median()),
            "Previous_Energy_Consumption": float(
                dataset["Previous_Energy_Consumption"].median()
            ),
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, name="building-simulator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval_seconds + 1)

    def _run(self) -> None:
        next_tick = time.monotonic()

        while not self._stopping.is_set():
            try:
                self._step()
            except Exception:
                LOGGER.exception("Building simulator update failed")

            next_tick += self.interval_seconds

            delay = next_tick - time.monotonic()

            if delay < 0:
                next_tick = time.monotonic()
                delay = 0

            if self._stopping.wait(delay):
                break

    def _bounded(self, column: str, value: float) -> float:
        minimum, maximum = self._limits[column]
        return float(np.clip(value, minimum, maximum))

    def _step(self) -> dict[str, Any]:
        now = datetime.now().astimezone()
        hour = now.hour
        is_weekend = int(now.weekday() >= 5)
        working_time = 8 <= hour < 18 and not is_weekend
        occupancy_target = 68 if working_time else (12 if 8 <= hour < 18 else 4)
        state = self._state

        with self._lock:
            occupancy = state["Occupancy"] + (
                occupancy_target - state["Occupancy"]
            ) * 0.08
            occupancy += self._rng.normal(0, 0.25)
            occupancy = self._bounded("Occupancy", occupancy)

            daily_temperature = 29 + 4 * np.sin(2 * np.pi * (hour - 8) / 24)
            temperature = state["Temperature"] + (
                daily_temperature - state["Temperature"]
            ) * 0.025 + self._rng.normal(0, 0.05)
            temperature = self._bounded("Temperature", temperature)
            humidity = self._bounded(
                "Humidity",
                state["Humidity"] + self._rng.normal(0, 0.25)
                + (62 - state["Humidity"]) * 0.05,
            )

            occupancy_ratio = occupancy / 100
            lighting_target = self._bounded(
                "Lighting_Load", 5 + occupancy_ratio * 82
            )
            lighting = self._bounded(
                "Lighting_Load", state["Lighting_Load"]
                + (lighting_target - state["Lighting_Load"]) * 0.2
                + self._rng.normal(0, 0.6),
            )
            computer_target = self._bounded(
                "Computer_Load", 2 + occupancy_ratio * 62
            )
            computer = self._bounded(
                "Computer_Load", state["Computer_Load"]
                + (computer_target - state["Computer_Load"]) * 0.2
                + self._rng.normal(0, 0.5),
            )
            hvac_demand = self._bounded(
                "HVAC_Load", 12 + abs(temperature - 24) * 5 + occupancy_ratio * 8
            )
            hvac = self._bounded(
                "HVAC_Load", state["HVAC_Load"]
                + (hvac_demand - state["HVAC_Load"]) * 0.15
                + self._rng.normal(0, 0.6)
            )
            other_target = self._bounded(
                "Other_Load", 14 + occupancy_ratio * 10
            )
            other = self._bounded(
                "Other_Load", state["Other_Load"]
                + (other_target - state["Other_Load"]) * 0.15
                + self._rng.normal(0, 0.4)
            )
            energy_target = self._bounded(
                "Previous_Energy_Consumption", 120 + occupancy * 2.6
                + lighting * 3.0 + hvac * 2.1 + computer * 2.1 + other * 1.8,
            )
            previous_energy = self._bounded(
                "Previous_Energy_Consumption", state["Previous_Energy_Consumption"]
                + (energy_target - state["Previous_Energy_Consumption"]) * 0.2,
            )
            state.update(
                {
                    "Date": now.strftime("%Y-%m-%d"),
                    "Hour": hour,
                    "Weekend": is_weekend,
                    "Holiday": 0,
                    "Temperature": temperature,
                    "Humidity": humidity,
                    "Occupancy": occupancy,
                    "Lighting_Load": lighting,
                    "HVAC_Load": hvac,
                    "Computer_Load": computer,
                    "Other_Load": other,
                    "Previous_Energy_Consumption": previous_energy,
                }
            )
            reading = {
                **state,
                "timestamp": now.isoformat(timespec="milliseconds"),
                "source": "simulated",
                "operating_state": (
                    "Working hours" if working_time else "After hours"
                ),
            }
            self._history.append(reading)
            self._last_step = time.monotonic()
            result = dict(reading)

        if self.on_update is not None:
            try:
                self.on_update(result)
            except Exception:
                LOGGER.exception("Live prediction callback failed")
        return result

    def current(self) -> dict[str, Any]:
        with self._lock:
            if self._history and time.monotonic() - self._last_step < self.interval_seconds:
                return dict(self._history[-1])
        return self._step()

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)[-limit:]
