"""
BioSCADA AI — canonical SCADA parameter definitions.

Single source of truth for the five monitored parameters (the "top-5"),
their control/alarm/trip bands, and their independent variables (PdM features).

Shared by: simulator, Flink jobs, ML training, agents, API.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Parameter:
    id: str
    name: str
    short: str
    unit: str
    asset: str
    # (low, high) bands — leaving control = breach
    control: Tuple[float, float]
    alarm: Tuple[float, float]
    trip: Tuple[float, float]
    baseline: float
    noise: float
    # independent variables (PdM features) driving this dependent variable
    features: List[str]
    root_cause: str
    model: str          # model family used in Phase 2
    sample_seconds: float

    @property
    def control_center(self) -> float:
        return (self.control[0] + self.control[1]) / 2

    def zone(self, value: float) -> str:
        """Return operating zone for a reading: control | alarm | trip."""
        if value < self.trip[0] or value > self.trip[1]:
            return "trip"
        if value < self.alarm[0] or value > self.alarm[1]:
            return "alarm"
        return "control"


PARAMETERS: Dict[str, Parameter] = {
    "temp": Parameter(
        id="temp",
        name="Reactor Temperature",
        short="Temperature",
        unit="degC",
        asset="BR-12",
        control=(36.5, 37.5),
        alarm=(36.0, 38.0),
        trip=(35.5, 38.5),
        baseline=37.0,
        noise=0.06,
        features=[
            "coolant_flow_rate",
            "jacket_temperature",
            "steam_valve_position",
            "heat_exchanger_dp",
            "agitator_torque",
        ],
        root_cause="Cooling-water flow fluctuation / heat-exchanger fouling",
        model="gbm",
        sample_seconds=1.0,
    ),
    "ph": Parameter(
        id="ph",
        name="pH",
        short="pH",
        unit="pH",
        asset="BR-12",
        control=(6.8, 7.2),
        alarm=(6.6, 7.4),
        trip=(6.4, 7.6),
        baseline=7.0,
        noise=0.04,
        features=[
            "acid_base_dose_rate",
            "co2_accumulation",
            "agitator_speed",
            "probe_drift_mv",
            "dissolved_oxygen",
        ],
        root_cause="Acid/base dosing lag / valve actuation delay",
        model="random_forest",
        sample_seconds=1.0,
    ),
    "press": Parameter(
        id="press",
        name="Differential Pressure",
        short="Pressure",
        unit="kPa",
        asset="FIL-07",
        control=(100.0, 110.0),
        alarm=(96.0, 114.0),
        trip=(92.0, 118.0),
        baseline=105.0,
        noise=0.7,
        features=[
            "filter_dp",
            "gas_exhaust_flow",
            "pump_speed_hz",
            "valve_position",
            "seal_integrity_index",
        ],
        root_cause="Filter blockage / gas-exhaust restriction",
        model="gbm",
        sample_seconds=2.0,
    ),
    "cond": Parameter(
        id="cond",
        name="Conductivity (WFI)",
        short="Conductivity",
        unit="uS/cm",
        asset="WFI-02",
        control=(700.0, 900.0),
        alarm=(640.0, 960.0),
        trip=(600.0, 1000.0),
        baseline=800.0,
        noise=9.0,
        features=[
            "water_flow_rate",
            "resin_bed_dp",
            "regeneration_cycle_count",
            "toc_level",
            "feed_composition",
        ],
        root_cause="Salt / nutrient feed composition error",
        model="svm",
        sample_seconds=5.0,
    ),
    "hum": Parameter(
        id="hum",
        name="Humidity (Cleanroom)",
        short="Humidity",
        unit="pct",
        asset="CR-A1",
        control=(40.0, 55.0),
        alarm=(36.0, 59.0),
        trip=(32.0, 63.0),
        baseline=47.0,
        noise=0.5,
        features=[
            "hvac_fan_speed",
            "cooling_coil_temp",
            "hepa_dp",
            "outdoor_humidity",
            "door_open_count",
        ],
        root_cause="HVAC coil / HEPA filter inefficiency",
        model="ann",
        sample_seconds=5.0,
    ),
}

PARAM_IDS: List[str] = list(PARAMETERS.keys())

# Molecules in the considered portfolio (DRL)
MOLECULES = ["Omez", "Nise", "Mintop"]

# Kafka topics
TOPIC_TELEMETRY = "scada.telemetry"
TOPIC_BREACH = "breach.events"
TOPIC_REMEDY = "remedy.actions"
TOPIC_AUDIT = "audit.trail"


def get(param_id: str) -> Parameter:
    if param_id not in PARAMETERS:
        raise KeyError(f"Unknown parameter '{param_id}'. Known: {PARAM_IDS}")
    return PARAMETERS[param_id]
