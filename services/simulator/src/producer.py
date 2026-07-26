"""
BioSCADA AI — Phase 0: SCADA telemetry simulator.

Emits realistic, physically-plausible sensor readings into Kafka topic
`scada.telemetry`, keyed by parameter id so each parameter lands in a
stable partition.

Each record carries BOTH:
  * the dependent variable (the SCADA parameter reading), and
  * its independent variables (the PdM features),
so that Flink can aggregate features and the model can score inline.

Breaches can be injected on demand (HTTP control plane) or occur naturally
via slow drift, which is what makes the downstream pipeline interesting.

Free stack: Apache Kafka (or Redpanda) + kafka-python.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import signal
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Optional

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
except ImportError:
    KafkaProducer = None  # type: ignore
    class NoBrokersAvailable(Exception): pass  # type: ignore

from parameters import PARAMETERS, PARAM_IDS, TOPIC_TELEMETRY, MOLECULES, Parameter

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s [simulator] %(message)s",
)
log = logging.getLogger("simulator")

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
BATCH_ID = os.getenv("BATCH_ID", "BR-12-2406")
MOLECULE = os.getenv("MOLECULE", MOLECULES[0])


@dataclass
class Excursion:
    """An injected or naturally-occurring drift away from setpoint."""
    target: float
    rise_ticks: int
    hold_ticks: int
    fall_ticks: int
    tick: int = 0

    def factor(self) -> float:
        """0..1 blend factor toward the excursion target."""
        t = self.tick
        if t < self.rise_ticks:
            return t / max(self.rise_ticks, 1)
        if t < self.rise_ticks + self.hold_ticks:
            return 1.0
        fall_t = t - self.rise_ticks - self.hold_ticks
        return max(0.0, 1.0 - fall_t / max(self.fall_ticks, 1))

    def done(self) -> bool:
        return self.tick >= self.rise_ticks + self.hold_ticks + self.fall_ticks


class ParameterSimulator:
    """Simulates one SCADA parameter and its correlated PdM features."""

    def __init__(self, param: Parameter, seed: Optional[int] = None):
        self.p = param
        self.rng = random.Random(seed)
        self.value = param.baseline
        self.excursion: Optional[Excursion] = None
        self.phase = self.rng.random() * math.tau  # for slow sinusoidal drift
        self.tick = 0

    def inject(self, severity: str = "alarm", direction: Optional[int] = None) -> None:
        """Inject a breach: severity in {alarm, trip}."""
        p = self.p
        d = direction if direction is not None else (1 if self.rng.random() > 0.5 else -1)
        if severity == "trip":
            lo, hi = p.trip
            target = hi + (hi - p.alarm[1]) * 0.4 if d > 0 else lo - (p.alarm[0] - lo) * 0.4
        else:
            target = (
                p.alarm[1] + (p.trip[1] - p.alarm[1]) * self.rng.uniform(0.25, 0.7)
                if d > 0
                else p.alarm[0] - (p.alarm[0] - p.trip[0]) * self.rng.uniform(0.25, 0.7)
            )
        self.excursion = Excursion(
            target=target, rise_ticks=8, hold_ticks=18, fall_ticks=14
        )
        log.info("Injected %s excursion on %s -> target %.2f", severity, p.id, target)

    def _maybe_natural_drift(self) -> None:
        """Occasionally start a mild natural excursion so the demo is alive."""
        if self.excursion is None and self.rng.random() < 0.0025:
            self.inject("alarm")

    def step(self) -> float:
        p = self.p
        self.tick += 1
        self._maybe_natural_drift()

        # slow sinusoidal wander so the trace never looks synthetic-flat
        wander = math.sin(self.tick / 90.0 + self.phase) * p.noise * 1.5
        target = p.baseline + wander

        if self.excursion is not None:
            f = self.excursion.factor()
            target = p.baseline + (self.excursion.target - p.baseline) * f
            self.excursion.tick += 1
            if self.excursion.done():
                self.excursion = None

        # first-order lag toward target + gaussian sensor noise
        self.value += (target - self.value) * 0.18
        self.value += self.rng.gauss(0, p.noise)
        # clamp to physically sane envelope
        lo, hi = p.trip[0] - 2 * p.noise * 10, p.trip[1] + 2 * p.noise * 10
        self.value = max(lo, min(hi, self.value))
        return self.value

    def features(self) -> Dict[str, float]:
        """
        Independent variables, correlated with the deviation of the dependent
        variable. This is what makes the PdM model learnable rather than noise.
        """
        p = self.p
        dev = (self.value - p.control_center) / max(
            (p.alarm[1] - p.alarm[0]) / 2, 1e-6
        )  # normalized deviation, ~[-1, 1]
        r = self.rng

        def corr(base: float, weight: float, scale: float) -> float:
            """Feature correlated with deviation plus its own noise."""
            return round(base + dev * weight * scale + r.gauss(0, scale * 0.12), 3)

        pid = p.id
        if pid == "temp":
            return {
                "coolant_flow_rate": corr(120.0, -1.0, 18.0),      # less coolant -> hotter
                "jacket_temperature": corr(32.0, 1.0, 4.0),
                "steam_valve_position": corr(28.0, 1.0, 9.0),
                "heat_exchanger_dp": corr(24.0, 1.0, 7.0),
                "agitator_torque": corr(48.0, 0.4, 6.0),
            }
        if pid == "ph":
            return {
                "acid_base_dose_rate": corr(1.8, -1.0, 0.6),
                "co2_accumulation": corr(4.2, 1.0, 1.1),
                "agitator_speed": corr(140.0, -0.3, 12.0),
                "probe_drift_mv": corr(3.0, 0.6, 4.0),
                "dissolved_oxygen": corr(38.0, -0.5, 6.0),
            }
        if pid == "press":
            return {
                "filter_dp": corr(28.0, 1.0, 9.0),
                "gas_exhaust_flow": corr(75.0, -1.0, 12.0),
                "pump_speed_hz": corr(44.0, 0.7, 5.0),
                "valve_position": corr(55.0, -0.6, 10.0),
                "seal_integrity_index": corr(0.94, -0.4, 0.05),
            }
        if pid == "cond":
            return {
                "water_flow_rate": corr(220.0, -0.8, 25.0),
                "resin_bed_dp": corr(31.0, 1.0, 8.0),
                "regeneration_cycle_count": corr(140.0, 0.9, 30.0),
                "toc_level": corr(0.32, 1.0, 0.12),
                "feed_composition": corr(1.00, 1.0, 0.09),
            }
        # humidity
        return {
            "hvac_fan_speed": corr(62.0, -1.0, 11.0),
            "cooling_coil_temp": corr(12.5, 1.0, 2.2),
            "hepa_dp": corr(210.0, 1.0, 26.0),
            "outdoor_humidity": corr(58.0, 0.5, 12.0),
            "door_open_count": corr(2.0, 0.8, 1.6),
        }

    def record(self) -> Dict:
        value = self.step()
        p = self.p
        return {
            "param": p.id,
            "asset": p.asset,
            "value": round(value, 4),
            "unit": p.unit,
            "zone": p.zone(value),
            "batch_id": BATCH_ID,
            "molecule": MOLECULE,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "features": self.features(),
        }


class SimulatorService:
    def __init__(self, bootstrap: str, rate_hz: float = 2.0, seed: Optional[int] = None):
        self.bootstrap = bootstrap
        self.rate_hz = rate_hz
        self.sims: Dict[str, ParameterSimulator] = {
            pid: ParameterSimulator(PARAMETERS[pid], seed=(seed + i) if seed else None)
            for i, pid in enumerate(PARAM_IDS)
        }
        self.producer: Optional[KafkaProducer] = None
        self._stop = threading.Event()
        self.sent = 0

    def connect(self, retries: int = 30, delay: float = 2.0) -> None:
        for attempt in range(1, retries + 1):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    acks="all",
                    linger_ms=20,
                    retries=5,
                )
                log.info("Connected to Kafka at %s", self.bootstrap)
                return
            except NoBrokersAvailable:
                log.warning("Kafka not ready (attempt %d/%d), retrying...", attempt, retries)
                time.sleep(delay)
        raise RuntimeError(f"Could not reach Kafka at {self.bootstrap}")

    def inject(self, param_id: str, severity: str = "alarm") -> bool:
        sim = self.sims.get(param_id)
        if not sim:
            return False
        sim.inject(severity)
        return True

    def run(self) -> None:
        assert self.producer is not None, "call connect() first"
        interval = 1.0 / self.rate_hz
        log.info("Producing to '%s' at %.1f Hz per parameter", TOPIC_TELEMETRY, self.rate_hz)
        while not self._stop.is_set():
            start = time.time()
            for pid, sim in self.sims.items():
                rec = sim.record()
                self.producer.send(TOPIC_TELEMETRY, key=pid, value=rec)
                self.sent += 1
                if rec["zone"] != "control":
                    log.info(
                        "%s %s=%.3f %s  ZONE=%s",
                        rec["asset"], pid, rec["value"], rec["unit"], rec["zone"].upper(),
                    )
            if self.sent % 200 == 0:
                self.producer.flush()
                log.info("Produced %d records", self.sent)
            time.sleep(max(0.0, interval - (time.time() - start)))

    def stop(self) -> None:
        self._stop.set()
        if self.producer:
            self.producer.flush()
            self.producer.close()
        log.info("Simulator stopped after %d records", self.sent)


def start_control_plane(svc: SimulatorService, port: int = 8081) -> None:
    """Tiny HTTP control plane so the frontend can inject breaches on demand."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse, parse_qs

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: Dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            if u.path == "/health":
                return self._json(200, {"status": "ok", "sent": svc.sent})
            if u.path == "/inject":
                q = parse_qs(u.query)
                pid = (q.get("param") or [""])[0]
                sev = (q.get("severity") or ["alarm"])[0]
                ok = svc.inject(pid, sev)
                return self._json(200 if ok else 404, {"injected": ok, "param": pid, "severity": sev})
            return self._json(404, {"error": "not found"})

        def log_message(self, *_):  # silence default noisy logging
            return

    srv = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("Control plane on :%d  (GET /inject?param=ph&severity=alarm)", port)


def main() -> None:
    ap = argparse.ArgumentParser(description="BioSCADA telemetry simulator")
    ap.add_argument("--bootstrap", default=BOOTSTRAP)
    ap.add_argument("--rate", type=float, default=float(os.getenv("RATE_HZ", "2.0")),
                    help="records per second per parameter")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--control-port", type=int, default=int(os.getenv("CONTROL_PORT", "8081")))
    args = ap.parse_args()

    svc = SimulatorService(args.bootstrap, rate_hz=args.rate, seed=args.seed)
    svc.connect()
    start_control_plane(svc, args.control_port)

    signal.signal(signal.SIGTERM, lambda *_: svc.stop())
    signal.signal(signal.SIGINT, lambda *_: svc.stop())
    try:
        svc.run()
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()


if __name__ == "__main__":
    main()
