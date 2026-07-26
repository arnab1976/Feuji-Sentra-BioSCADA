"""
BioSCADA AI — Multi-agent layer with RAG.

Architecture
------------
    Orchestrator  ── RBAC-gated routing ──>  ParameterAgent (x5)
                                                  |
                                                  v
                                        semantic decomposition
                                        (sub-questions -> retrieval
                                         -> fusion -> synthesis)
                                                  |
                                                  v
                                          HumanEscalationAgent

This is the SEMANTIC decomposition half of the system. Flink already did
the RELATIONAL-PLAN decomposition upstream and handed us a structured
"breach + why" event; here the LLM decomposes the *question*.

Framework-neutral by design: the orchestration is plain Python, so it runs
without LangChain/LangGraph. A Haystack `Agent` + `ComponentTool` adapter is
provided for teams that prefer that route.

LLM: local by default (Ollama / llama.cpp) so no data leaves the network —
which is what makes this defensible in a GxP context.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s [agents] %(message)s",
)
log = logging.getLogger("agents")

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "services/simulator/src"))
sys.path.insert(0, str(_ROOT / "services/rag/src"))

from parameters import PARAMETERS, Parameter  # noqa: E402
from knowledge_base import KnowledgeBase, Chunk  # noqa: E402

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
INDEX_DIR = Path(os.getenv("RAG_INDEX", _ROOT / "services/rag/index"))


# =====================================================================
# LLM client — local-first, with a deterministic template fallback
# =====================================================================
class LLMClient:
    """
    Thin client for a locally-served open-weights model.

    Works with Ollama out of the box (free, self-hosted). If no server is
    reachable, falls back to a deterministic template composer so the whole
    pipeline still runs and is testable in CI / air-gapped environments.
    """

    def __init__(self, base_url: str = OLLAMA_URL, model: str = LLM_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2) as r:
                ok = r.status == 200
            if ok:
                log.info("LLM available: %s @ %s", self.model, self.base_url)
            return ok
        except Exception:
            log.warning("No LLM server at %s -> deterministic fallback composer",
                        self.base_url)
            return False

    def generate(self, prompt: str, system: str = "", temperature: float = 0.1,
                 max_tokens: int = 700) -> str:
        if not self.available:
            return ""
        try:
            import urllib.request
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/generate", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read()).get("response", "").strip()
        except Exception as exc:
            log.warning("LLM generation failed (%s)", exc)
            return ""


# =====================================================================
# Domain objects
# =====================================================================
@dataclass
class BreachEvent:
    """The structured payload Flink emits — input to the agent layer."""
    event_id: str
    param: str
    asset: str
    v_avg: float
    v_std: float = 0.0
    v_delta: float = 0.0
    zone: str = "alarm"
    p_breach: float = 0.0
    top_driver: str = ""
    batch_id: str = ""
    window_start: str = ""

    @classmethod
    def from_json(cls, raw: Dict) -> "BreachEvent":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class SubQuestion:
    intent: str          # sop | capa | oem | maintenance | history
    question: str
    source_types: List[str]


@dataclass
class RemedyStep:
    order: int
    action: str
    citation: str


@dataclass
class RemedyCard:
    event_id: str
    param: str
    asset: str
    root_cause: str
    steps: List[RemedyStep]
    citations: List[str]
    requires_esign: bool
    esign_reason: str
    confidence: float
    sub_questions: List[str] = field(default_factory=list)
    retrieved_chunks: int = 0
    generated_by: str = "template"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


# =====================================================================
# Parameter agent — one per SCADA parameter
# =====================================================================
class ParameterAgent:
    """
    Owns one dependent variable. Performs the semantic decomposition:
      1. plan sub-questions from the structured event
      2. retrieve per sub-question (multi-retrieval fan-out)
      3. fuse + re-rank
      4. synthesise a cited remedy
    """

    def __init__(self, param: Parameter, kb: KnowledgeBase, llm: LLMClient):
        self.param = param
        self.kb = kb
        self.llm = llm
        self.name = f"{param.short} Agent"
        self.role = f"agent.execute.{param.id}"

    # ---------- step 1: query planning / sub-question split ----------
    def plan(self, event: BreachEvent) -> List[SubQuestion]:
        p = self.param
        direction = "above" if event.v_avg > p.control[1] else "below"
        return [
            SubQuestion(
                intent="sop",
                question=(f"What is the approved corrective procedure when {p.name} on "
                          f"{p.asset} goes {direction} its control band of "
                          f"{p.control[0]}-{p.control[1]} {p.unit}?"),
                source_types=["sop"],
            ),
            SubQuestion(
                intent="capa",
                question=(f"What prior CAPA records exist for {p.name} excursions on "
                          f"{p.asset}, and which corrective actions were effective?"),
                source_types=["capa"],
            ),
            SubQuestion(
                intent="oem",
                question=(f"What do the OEM manual and maintenance history say about "
                          f"{event.top_driver or p.features[0]} affecting {p.name}?"),
                source_types=["oem", "maintenance"],
            ),
        ]

    # ---------- step 2+3: multi-retrieval, fusion, re-rank ----------
    def retrieve(self, subs: List[SubQuestion], top_k: int = 4
                 ) -> Tuple[List[Dict], List[Chunk]]:
        fused: Dict[str, Dict] = {}
        for sq in subs:
            hits = self.kb.search(
                sq.question, top_k=top_k, param=self.param.id,
                source_types=sq.source_types, use_mmr=True,
            )
            if not hits:  # fall back to unfiltered search if the filter is too tight
                hits = self.kb.search(sq.question, top_k=top_k, use_mmr=True)
            for rank, h in enumerate(hits):
                c: Chunk = h["chunk"]
                # Reciprocal Rank Fusion across sub-questions
                rrf = 1.0 / (60 + rank)
                if c.id in fused:
                    fused[c.id]["score"] += rrf
                else:
                    fused[c.id] = {"chunk": c, "score": rrf, "intent": sq.intent}
        ranked = sorted(fused.values(), key=lambda x: -x["score"])
        return ranked, [r["chunk"] for r in ranked]

    # ---------- step 4: synthesis ----------
    def synthesize(self, event: BreachEvent, ranked: List[Dict]) -> RemedyCard:
        p = self.param
        context = "\n\n".join(
            f"[{r['chunk'].citation()}] ({r['chunk'].source_type})\n{r['chunk'].text}"
            for r in ranked[:6]
        )
        citations = list(dict.fromkeys(r["chunk"].citation() for r in ranked[:6]))

        requires_esign, reason = self.esign_policy(event)
        steps: List[RemedyStep] = []
        generated_by = "template"

        if self.llm.available and context:
            system = (
                "You are a GxP-compliant pharmaceutical manufacturing assistant. "
                "Answer ONLY from the provided approved documents. Cite the source "
                "identifier in square brackets after each step. If the documents do "
                "not cover the situation, say so explicitly. Never invent a procedure."
            )
            prompt = (
                f"APPROVED DOCUMENTS:\n{context}\n\n"
                f"SITUATION:\n"
                f"- Parameter: {p.name} on {p.asset}\n"
                f"- Reading: {event.v_avg} {p.unit} (control band {p.control[0]}-{p.control[1]})\n"
                f"- Zone: {event.zone.upper()}\n"
                f"- P(breach): {event.p_breach}\n"
                f"- Leading driver: {event.top_driver}\n\n"
                "Give the corrective procedure as a numbered list of at most 5 concrete "
                "steps, each ending with its citation in square brackets."
            )
            out = self.llm.generate(prompt, system=system)
            if out:
                steps = self._parse_steps(out)
                generated_by = self.llm.model

        if not steps:
            steps = self._template_steps(event, ranked)

        confidence = min(0.98, 0.45 + 0.09 * len(ranked)) if ranked else 0.2
        return RemedyCard(
            event_id=event.event_id,
            param=p.id,
            asset=event.asset or p.asset,
            root_cause=p.root_cause,
            steps=steps,
            citations=citations,
            requires_esign=requires_esign,
            esign_reason=reason,
            confidence=round(confidence, 3),
            sub_questions=[s.question for s in self.plan(event)],
            retrieved_chunks=len(ranked),
            generated_by=generated_by,
        )

    @staticmethod
    def _parse_steps(text: str) -> List[RemedyStep]:
        steps: List[RemedyStep] = []
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"^(\d+)[.)]\s*(.+)$", line)
            if not m:
                continue
            body = m.group(2).strip()
            cites = re.findall(r"\[([^\]]+)\]", body)
            action = re.sub(r"\s*\[[^\]]+\]", "", body).strip()
            steps.append(RemedyStep(order=int(m.group(1)), action=action,
                                    citation=", ".join(cites) if cites else ""))
        return steps[:5]

    def _template_steps(self, event: BreachEvent, ranked: List[Dict]) -> List[RemedyStep]:
        """
        Deterministic grounded fallback: lift imperative 'Step N.' sentences
        straight out of the retrieved SOP text. Still fully cited — it just
        does not paraphrase.
        """
        steps: List[RemedyStep] = []
        seen: set = set()   # chunk overlap re-emits the same step; dedupe on content
        for r in ranked:
            c: Chunk = r["chunk"]
            if c.source_type != "sop":
                continue
            for sent in re.findall(r"Step\s+\d+\.\s*([^\n]+?)(?=\s*Step\s+\d+\.|\Z)",
                                   c.text, flags=re.S):
                clean = " ".join(sent.split())
                if not (20 < len(clean) < 300):
                    continue
                key = clean[:60].lower()
                if key in seen:
                    continue
                seen.add(key)
                steps.append(RemedyStep(order=len(steps) + 1,
                                        action=clean, citation=c.citation()))
                if len(steps) >= 4:
                    break
            if len(steps) >= 4:
                break
        if not steps:
            p = self.param
            steps = [RemedyStep(
                order=1,
                action=(f"No approved procedure retrieved for this {p.name} condition. "
                        f"Escalate to QA and record a deviation."),
                citation="ESCALATION-DEFAULT")]
        return steps

    # ---------- GxP policy ----------
    def esign_policy(self, event: BreachEvent) -> Tuple[bool, str]:
        """
        Encodes the documented GxP rules. Deliberately conservative: this is
        the gate between an autonomous suggestion and a real control action.
        """
        if self.param.id == "ph":
            return True, "SOP-PH-009: all pH setpoint changes require QA e-signature"
        if event.zone == "trip":
            return True, "Trip-zone excursion — QA review obligation"
        if event.p_breach >= 0.80:
            return True, "High predicted breach probability (>=0.80)"
        return False, "Micro-adjustment within policy — operator signature captured for traceability"

    # ---------- full run ----------
    def run(self, event: BreachEvent) -> RemedyCard:
        log.info("[%s] activated for %s (zone=%s, p=%.2f)",
                 self.name, event.event_id, event.zone, event.p_breach)
        subs = self.plan(event)
        ranked, _ = self.retrieve(subs)
        log.info("[%s] %d sub-questions -> %d fused chunks",
                 self.name, len(subs), len(ranked))
        return self.synthesize(event, ranked)


# =====================================================================
# Human escalation agent
# =====================================================================
@dataclass
class Signature:
    signer: str
    role: str
    reason: str
    signed_at: str
    committed: bool


class HumanEscalationAgent:
    """
    The single human-in-the-loop authority (QA / Production).
    All five parameter agents route here when policy demands a signature.
    """
    name = "QA / Production Escalation Agent"

    def __init__(self):
        self.pending: Dict[str, RemedyCard] = {}
        self.signatures: Dict[str, Signature] = {}

    def submit(self, card: RemedyCard) -> str:
        self.pending[card.event_id] = card
        log.info("[escalation] %s queued for signature (%s)",
                 card.event_id, card.esign_reason)
        return card.event_id

    def sign(self, event_id: str, signer: str, role: str, reason: str) -> Signature:
        if event_id not in self.pending:
            raise KeyError(f"No pending action for {event_id}")
        if len(signer.strip()) < 3:
            raise ValueError("Signer name is required (21 CFR Part 11)")
        if len(reason.strip()) < 4:
            raise ValueError("Reason for signature is required (21 CFR Part 11)")
        sig = Signature(signer=signer.strip(), role=role, reason=reason.strip(),
                        signed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        committed=True)
        self.signatures[event_id] = sig
        self.pending.pop(event_id, None)
        log.info("[escalation] %s SIGNED by %s (%s)", event_id, signer, role)
        return sig

    def reject(self, event_id: str, signer: str, reason: str) -> Signature:
        sig = Signature(signer=signer, role="Rejected", reason=reason,
                        signed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        committed=False)
        self.signatures[event_id] = sig
        self.pending.pop(event_id, None)
        log.info("[escalation] %s REJECTED by %s", event_id, signer)
        return sig


# =====================================================================
# Orchestrator — RBAC-gated routing
# =====================================================================
class Orchestrator:
    """
    Routes a breach event to exactly the agent that owns that parameter.
    Any other agent is denied — this is the RBAC boundary, enforced before
    the request ever reaches the model.
    """

    def __init__(self, kb: KnowledgeBase, llm: Optional[LLMClient] = None):
        self.kb = kb
        self.llm = llm or LLMClient()
        self.agents: Dict[str, ParameterAgent] = {
            pid: ParameterAgent(p, kb, self.llm) for pid, p in PARAMETERS.items()
        }
        self.escalation = HumanEscalationAgent()

    def authorize(self, agent_id: str, event: BreachEvent) -> bool:
        return agent_id == event.param

    def dispatch(self, event: BreachEvent) -> RemedyCard:
        agent = self.agents.get(event.param)
        if agent is None:
            raise KeyError(f"No agent owns parameter '{event.param}'")
        if not self.authorize(agent.param.id, event):
            raise PermissionError("RBAC denied: agent/parameter mismatch")
        card = agent.run(event)
        if card.requires_esign:
            self.escalation.submit(card)
        return card


# =====================================================================
# Optional Haystack adapter (no LangChain / LangGraph required)
# =====================================================================
def build_haystack_pipeline(kb: KnowledgeBase):  # pragma: no cover - optional
    """
    Equivalent wiring using Haystack, for teams standardising on it.
    Each ParameterAgent becomes a ComponentTool of a coordinator Agent.
    Requires: pip install haystack-ai
    """
    try:
        from haystack.components.agents import Agent
        from haystack.tools import ComponentTool
        from haystack.components.generators.chat import OpenAIChatGenerator
    except ImportError as exc:
        raise RuntimeError("pip install haystack-ai to use the Haystack adapter") from exc
    raise NotImplementedError(
        "Adapter scaffold: wrap each ParameterAgent.run as a ComponentTool and "
        "register with a coordinator Agent. See docs/ARCHITECTURE.md."
    )


def load_orchestrator(index_dir: Path = INDEX_DIR) -> Orchestrator:
    kb = (KnowledgeBase.load(index_dir) if (index_dir / "chunks.json").exists()
          else _build_kb_fresh())
    return Orchestrator(kb)


def _build_kb_fresh() -> KnowledgeBase:
    from knowledge_base import build_from_data_dir
    log.info("No prebuilt index; building knowledge base from data/")
    return build_from_data_dir(use_qdrant=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run one agent cycle")
    ap.add_argument("--param", default="temp", choices=list(PARAMETERS))
    ap.add_argument("--value", type=float, default=None)
    ap.add_argument("--zone", default="alarm")
    ap.add_argument("--p-breach", type=float, default=0.72)
    a = ap.parse_args()

    orch = load_orchestrator()
    p = PARAMETERS[a.param]
    value = a.value if a.value is not None else p.alarm[1] + 0.2
    evt = BreachEvent(
        event_id=f"evt-{a.param}-demo", param=a.param, asset=p.asset,
        v_avg=value, v_std=0.08, v_delta=0.4, zone=a.zone,
        p_breach=a.p_breach, top_driver=p.features[0], batch_id="BR-12-2406",
    )
    card = orch.dispatch(evt)
    print("\n" + "=" * 72)
    print(json.dumps(card.to_dict(), indent=2)[:2200])
    print("=" * 72)
