"""
BioSCADA AI — pipeline tests.

Covers every phase: parameter model, simulator physics, Flink UDF scoring,
knowledge-base ingestion & retrieval, agent decomposition, and the GxP
policy decision table.

Run:  pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("services/simulator/src", "services/rag/src", "services/agents/src"):
    sys.path.insert(0, str(ROOT / sub))

from parameters import PARAMETERS, PARAM_IDS  # noqa: E402


# =====================================================================
# Parameters
# =====================================================================
class TestParameters:
    def test_all_five_present(self):
        assert set(PARAM_IDS) == {"temp", "ph", "press", "cond", "hum"}

    @pytest.mark.parametrize("pid", PARAM_IDS)
    def test_bands_are_nested(self, pid):
        """trip must enclose alarm, which must enclose control."""
        p = PARAMETERS[pid]
        assert p.trip[0] <= p.alarm[0] < p.control[0]
        assert p.control[1] < p.alarm[1] <= p.trip[1]

    @pytest.mark.parametrize("pid", PARAM_IDS)
    def test_zone_classification(self, pid):
        p = PARAMETERS[pid]
        assert p.zone(p.control_center) == "control"
        assert p.zone(p.alarm[1] + (p.trip[1] - p.alarm[1]) / 2) == "alarm"
        assert p.zone(p.trip[1] + 1) == "trip"

    @pytest.mark.parametrize("pid", PARAM_IDS)
    def test_each_has_five_independent_variables(self, pid):
        assert len(PARAMETERS[pid].features) == 5


# =====================================================================
# Phase 0 — simulator
# =====================================================================
class TestSimulator:
    def test_baseline_stays_in_control_band(self):
        from producer import ParameterSimulator
        sim = ParameterSimulator(PARAMETERS["temp"], seed=7)
        zones = [sim.record()["zone"] for _ in range(60)]
        assert zones.count("control") > 45, "baseline should mostly sit in band"

    def test_injection_reaches_alarm(self):
        from producer import ParameterSimulator
        sim = ParameterSimulator(PARAMETERS["temp"], seed=7)
        sim.inject("alarm", direction=1)
        zones = [sim.record()["zone"] for _ in range(40)]
        assert "alarm" in zones

    def test_features_correlate_with_deviation(self):
        """Coolant flow must fall as temperature rises — the physics the model learns."""
        from producer import ParameterSimulator
        sim = ParameterSimulator(PARAMETERS["temp"], seed=11)
        base = [sim.record() for _ in range(10)][-1]["features"]["coolant_flow_rate"]
        sim.inject("alarm", direction=1)
        for _ in range(25):
            rec = sim.record()
        assert rec["features"]["coolant_flow_rate"] < base

    def test_record_schema(self):
        from producer import ParameterSimulator
        rec = ParameterSimulator(PARAMETERS["ph"], seed=3).record()
        for key in ("param", "asset", "value", "unit", "zone", "ts", "features"):
            assert key in rec


# =====================================================================
# Phase 1 — Flink scoring UDF
# =====================================================================
class TestFlinkScoring:
    @staticmethod
    def _score(param, v_avg, v_std=0.05, v_delta=0.2):
        sys.path.insert(0, str(ROOT / "services/flink-jobs/src"))
        import types
        src = (ROOT / "services/flink-jobs/src/breach_detection_job.py").read_text()
        src = src.replace("from pyflink.table import EnvironmentSettings, TableEnvironment, DataTypes", "")
        src = src.replace("from pyflink.table.udf import udf", "")
        src = src.replace("@udf(result_type=DataTypes.DOUBLE())", "")
        src = src.replace("@udf(result_type=DataTypes.STRING())", "")
        src = src.split("def build_environment")[0]
        mod = types.ModuleType("flinkmod")
        exec(compile(src, "flinkmod", "exec"), mod.__dict__)
        return mod.pdm_score(param, v_avg, v_std, v_delta)

    def test_score_bounded(self):
        assert 0.0 <= self._score("temp", 999) <= 1.0
        assert 0.0 <= self._score("temp", 37.0) <= 1.0

    def test_score_monotonic_in_deviation(self):
        p = PARAMETERS["temp"]
        vals = [p.control_center, 37.4, 37.9, 38.3]
        scores = [self._score("temp", v) for v in vals]
        assert scores == sorted(scores)

    def test_in_band_scores_low(self):
        assert self._score("temp", PARAMETERS["temp"].control_center) < 0.1

    def test_alarm_scores_high(self):
        assert self._score("temp", 38.3) > 0.6

    def test_volatility_increases_score(self):
        assert self._score("temp", 37.8, v_std=0.4) > self._score("temp", 37.8, v_std=0.02)


# =====================================================================
# RAG knowledge base
# =====================================================================
@pytest.fixture(scope="module")
def kb():
    from knowledge_base import build_from_data_dir
    return build_from_data_dir(ROOT / "data", use_qdrant=False)


class TestKnowledgeBase:
    def test_ingests_both_families(self, kb):
        assert any(c.structured for c in kb.chunks), "expected structured records"
        assert any(not c.structured for c in kb.chunks), "expected unstructured docs"

    def test_source_types_correct(self, kb):
        types_ = {c.source_type for c in kb.chunks}
        assert {"sop", "capa", "maintenance", "batch", "oem"} <= types_

    def test_maintenance_not_mislabelled_as_capa(self, kb):
        """Regression: maintenance_log.csv lives in capa/ but is not a CAPA."""
        maint = [c for c in kb.chunks if c.source_type == "maintenance"]
        assert maint, "maintenance log should be detected by filename"
        assert all("None" not in c.text[:100] for c in maint), "wrong verbalizer applied"

    def test_no_none_filled_chunks(self, kb):
        assert not [c for c in kb.chunks if "None" in c.text[:120]]

    def test_retrieval_finds_relevant_sop(self, kb):
        hits = kb.search("reactor temperature rising reduce coolant setpoint", top_k=5)
        assert hits
        assert any("THM-014" in h["chunk"].source_id for h in hits)

    def test_retrieval_respects_param_filter(self, kb):
        hits = kb.search("excursion response", top_k=5, param="ph")
        assert all(h["chunk"].param in (None, "ph") for h in hits)

    def test_chunks_carry_citations(self, kb):
        assert all(c.citation() for c in kb.chunks)


# =====================================================================
# Agents
# =====================================================================
@pytest.fixture(scope="module")
def orchestrator(kb):
    from agents import Orchestrator
    return Orchestrator(kb)


class TestAgents:
    def _event(self, pid, value=None, zone="alarm", p=0.7):
        from agents import BreachEvent
        par = PARAMETERS[pid]
        return BreachEvent(
            event_id=f"evt-{pid}-test", param=pid, asset=par.asset,
            v_avg=value if value is not None else par.alarm[1] + 0.2,
            zone=zone, p_breach=p, top_driver=par.features[0])

    def test_one_agent_per_parameter(self, orchestrator):
        assert set(orchestrator.agents) == set(PARAM_IDS)

    def test_rbac_blocks_mismatched_agent(self, orchestrator):
        evt = self._event("temp")
        assert orchestrator.authorize("temp", evt) is True
        assert orchestrator.authorize("ph", evt) is False

    def test_decomposes_into_sub_questions(self, orchestrator):
        agent = orchestrator.agents["temp"]
        subs = agent.plan(self._event("temp"))
        assert len(subs) >= 3
        assert {s.intent for s in subs} >= {"sop", "capa"}

    def test_remedy_card_is_cited(self, orchestrator):
        card = orchestrator.dispatch(self._event("temp"))
        assert card.steps, "expected remedy steps"
        assert card.citations, "remedy must cite sources"
        assert card.retrieved_chunks > 0

    def test_no_duplicate_steps(self, orchestrator):
        """Regression: overlapping chunks used to re-emit the same step."""
        card = orchestrator.dispatch(self._event("temp"))
        actions = [s.action for s in card.steps]
        assert len(actions) == len(set(actions))

    def test_ph_always_requires_esign(self, orchestrator):
        card = orchestrator.dispatch(self._event("ph", p=0.2))
        assert card.requires_esign is True
        assert "PH-009" in card.esign_reason

    def test_low_risk_temp_is_autonomous(self, orchestrator):
        card = orchestrator.dispatch(self._event("temp", p=0.4, zone="alarm"))
        assert card.requires_esign is False

    def test_trip_zone_escalates(self, orchestrator):
        card = orchestrator.dispatch(self._event("cond", zone="trip", p=0.4))
        assert card.requires_esign is True

    def test_high_probability_escalates(self, orchestrator):
        card = orchestrator.dispatch(self._event("hum", zone="alarm", p=0.92))
        assert card.requires_esign is True


# =====================================================================
# Human escalation / e-signature
# =====================================================================
class TestEscalation:
    def test_signature_requires_name_and_reason(self, orchestrator):
        from agents import BreachEvent
        evt = BreachEvent(event_id="evt-sign-test", param="ph", asset="BR-12",
                          v_avg=7.45, zone="alarm", p_breach=0.5)
        orchestrator.dispatch(evt)
        esc = orchestrator.escalation
        with pytest.raises(ValueError):
            esc.sign("evt-sign-test", "X", "QA", "ok")
        with pytest.raises(ValueError):
            esc.sign("evt-sign-test", "Valid Name", "QA", "no")
        sig = esc.sign("evt-sign-test", "Valid Name", "QA", "Approved within policy")
        assert sig.committed is True

    def test_unknown_event_rejected(self, orchestrator):
        with pytest.raises(KeyError):
            orchestrator.escalation.sign("does-not-exist", "A Name", "QA", "reason")


# =====================================================================
# Policy decision table (mirrors infra/config/opa/bioscada.rego)
# =====================================================================
class TestPolicyTable:
    @staticmethod
    def decide(param, zone, p_breach, roles):
        if f"agent.execute.{param}" not in roles:
            return "DENY"
        esign = param == "ph" or zone == "trip" or p_breach >= 0.80
        return "REQUIRE_ESIGN" if esign else "ALLOW"

    @pytest.mark.parametrize("param,zone,p,roles,expected", [
        ("ph",   "alarm", 0.40, ["agent.execute.ph"],   "REQUIRE_ESIGN"),
        ("temp", "alarm", 0.50, ["agent.execute.temp"], "ALLOW"),
        ("temp", "trip",  0.50, ["agent.execute.temp"], "REQUIRE_ESIGN"),
        ("cond", "alarm", 0.91, ["agent.execute.cond"], "REQUIRE_ESIGN"),
        ("hum",  "alarm", 0.80, ["agent.execute.hum"],  "REQUIRE_ESIGN"),
        ("hum",  "alarm", 0.79, ["agent.execute.hum"],  "ALLOW"),
        ("temp", "alarm", 0.50, ["agent.execute.ph"],   "DENY"),
    ])
    def test_decision_table(self, param, zone, p, roles, expected):
        assert self.decide(param, zone, p, roles) == expected


# =====================================================================
# Audit hash chain (mirrors infra/config/postgres-init.sql)
# =====================================================================
class TestAuditChain:
    @staticmethod
    def chain(entries):
        import hashlib, json
        prev, out = "GENESIS", []
        for ts, action, detail in entries:
            h = hashlib.sha256((prev + ts + action + json.dumps(detail)).encode()).hexdigest()
            out.append(h)
            prev = h
        return out

    def test_chain_is_deterministic(self):
        e = [("t1", "a", {"x": 1}), ("t2", "b", {"y": 2})]
        assert self.chain(e) == self.chain(e)

    def test_tamper_breaks_chain(self):
        good = self.chain([("t1", "a", {"x": 1}), ("t2", "b", {"y": 2})])
        bad = self.chain([("t1", "a", {"x": 999}), ("t2", "b", {"y": 2})])
        assert good != bad, "modifying an earlier row must invalidate later hashes"
