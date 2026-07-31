"""
BioSCADA AI — RAG knowledge base.

Consolidates BOTH knowledge families into one governed, citable corpus:

  UNSTRUCTURED : SOPs, OEM manuals, deviation reports  (markdown / PDF / text)
  STRUCTURED   : CAPA records, maintenance logs, batch records (CSV / JSON / SQL)

Structured rows are *verbalized* into natural-language chunks before
embedding, so a single hybrid retriever can serve both families while each
chunk keeps its provenance metadata for citation and GxP traceability.

Free stack:
    Embeddings  : sentence-transformers (BAAI/bge-small-en-v1.5)
    Vector store: Qdrant  (falls back to an in-process NumPy index)
    Keyword     : rank_bm25  (lexical half of hybrid retrieval)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s [rag-kb] %(message)s",
)
log = logging.getLogger("rag-kb")

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[3] / "data"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION = os.getenv("QDRANT_COLLECTION", "bioscada_kb")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "220"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))


# =====================================================================
# Chunk model — provenance travels with every chunk (needed for citations)
# =====================================================================
@dataclass
class Chunk:
    id: str
    text: str
    source_type: str          # sop | capa | oem | maintenance | batch | deviation
    source_id: str            # e.g. "SOP-THM-014 r6"
    title: str
    param: Optional[str] = None      # which SCADA parameter it pertains to
    asset: Optional[str] = None
    effective_date: Optional[str] = None
    structured: bool = False
    extra: Dict = field(default_factory=dict)

    def citation(self) -> str:
        return self.source_id or self.title


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode()).hexdigest()[:16]


def chunk_text(text: str, size: int = CHUNK_TOKENS,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Recursive-ish splitter: prefer paragraph boundaries, fall back to word
    windows. Keeps procedural steps intact, which matters for SOPs where a
    truncated step is worse than no step.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    count = 0
    for p in paras:
        n = len(p.split())
        if count + n > size and buf:
            chunks.append("\n\n".join(buf))
            # carry overlap
            tail = " ".join(" ".join(buf).split()[-overlap:]) if overlap else ""
            buf = [tail] if tail else []
            count = len(tail.split())
        buf.append(p)
        count += n
    if buf:
        chunks.append("\n\n".join(buf))

    # hard-split any oversized chunk
    final: List[str] = []
    for c in chunks:
        words = c.split()
        if len(words) <= size * 1.5:
            final.append(c)
        else:
            step = size - overlap
            for i in range(0, len(words), step):
                final.append(" ".join(words[i:i + size]))
    return [c for c in final if c.strip()]


# =====================================================================
# UNSTRUCTURED loaders
# =====================================================================
def load_unstructured(root: Path) -> List[Chunk]:
    """Load SOPs / OEM manuals / deviation reports from md, txt, pdf."""
    chunks: List[Chunk] = []
    if not root.exists():
        log.warning("Unstructured dir missing: %s", root)
        return chunks

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        text = _read_any(path)
        if not text.strip():
            continue
        meta = _parse_front_matter(text)
        body = meta.pop("_body", text)
        source_type = meta.get("type") or _infer_type(path)
        source_id = meta.get("id") or path.stem
        title = meta.get("title") or path.stem.replace("_", " ")

        for i, c in enumerate(chunk_text(body)):
            chunks.append(Chunk(
                id=_hash_id(str(path), str(i)),
                text=c,
                source_type=source_type,
                source_id=source_id,
                title=title,
                param=meta.get("param"),
                asset=meta.get("asset"),
                effective_date=meta.get("effective"),
                structured=False,
                extra={"path": str(path.relative_to(root.parent)), "chunk": i},
            ))
    log.info("Unstructured: %d chunks", len(chunks))
    return chunks


def _read_any(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # PyMuPDF
            with fitz.open(path) as doc:
                return "\n\n".join(page.get_text() for page in doc)
        except ImportError:
            log.warning("PyMuPDF not installed; skipping %s", path.name)
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_front_matter(text: str) -> Dict:
    """Minimal YAML-ish front matter parser (no external dependency)."""
    if not text.startswith("---"):
        return {"_body": text}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {"_body": text}
    meta: Dict = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"\'')
    meta["_body"] = parts[2].strip()
    return meta


def _infer_type(path: Path) -> str:
    """
    Infer the source type.

    Filename wins over directory: `capa/maintenance_log.csv` is a maintenance
    log that merely lives beside the CAPA records, and must not be verbalized
    with the CAPA template (that silently produces None-filled chunks).
    """
    keys = ("maintenance", "deviation", "batch", "capa", "sop", "oem")
    stem = path.stem.lower()
    for key in keys:
        if key in stem:
            return key
    parent = str(path.parent).lower()
    for key in keys:
        if key in parent:
            return key
    return "document"


# =====================================================================
# STRUCTURED loaders — rows are verbalized into retrievable prose
# =====================================================================
def verbalize_capa(row: Dict) -> str:
    return (
        f"CAPA record {row.get('capa_id')} for {row.get('parameter')} on asset "
        f"{row.get('asset')}, opened {row.get('opened')}. "
        f"Problem: {row.get('problem')} "
        f"Root cause: {row.get('root_cause')} "
        f"Corrective action taken: {row.get('corrective_action')} "
        f"Effectiveness: {row.get('effectiveness')}. "
        f"Recurrence after action: {row.get('recurrence', 'none recorded')}."
    )


def verbalize_maintenance(row: Dict) -> str:
    return (
        f"Maintenance work order {row.get('wo_id')} on {row.get('asset')} "
        f"({row.get('component')}) completed {row.get('completed')}. "
        f"Trigger: {row.get('trigger')}. Action: {row.get('action')}. "
        f"Findings: {row.get('findings')}. "
        f"Downtime: {row.get('downtime_min')} minutes. "
        f"Technician note: {row.get('note', '-')}"
    )


def verbalize_batch(row: Dict) -> str:
    return (
        f"Batch record {row.get('batch_id')} for molecule {row.get('molecule')}, "
        f"phase {row.get('phase')}. Parameter {row.get('parameter')} held "
        f"{row.get('setpoint')} with observed range {row.get('observed_range')}. "
        f"Deviations: {row.get('deviations', 'none')}. "
        f"Disposition: {row.get('disposition')}."
    )


VERBALIZERS = {
    "capa": (verbalize_capa, "capa_id"),
    "maintenance": (verbalize_maintenance, "wo_id"),
    "batch": (verbalize_batch, "batch_id"),
}


def load_structured(root: Path) -> List[Chunk]:
    """
    Load CSV/JSON structured records and verbalize each row.

    Why verbalize: embedding a raw CSV row retrieves poorly — the model has
    no linguistic signal. A verbalized sentence retrieves well AND keeps the
    original row in metadata for exact display.
    """
    chunks: List[Chunk] = []
    if not root.exists():
        log.warning("Structured dir missing: %s", root)
        return chunks

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
            continue
        kind = _infer_type(path)
        verbalizer, id_key = VERBALIZERS.get(kind, (None, None))
        rows = _read_rows(path)
        for i, row in enumerate(rows):
            text = verbalizer(row) if verbalizer else json.dumps(row, ensure_ascii=False)
            sid = str(row.get(id_key) if id_key else f"{path.stem}-{i}")
            chunks.append(Chunk(
                id=_hash_id(str(path), sid, str(i)),
                text=text,
                source_type=kind,
                source_id=sid,
                title=f"{kind.upper()} {sid}",
                param=row.get("parameter") or row.get("param"),
                asset=row.get("asset"),
                effective_date=row.get("completed") or row.get("opened"),
                structured=True,
                extra={"row": row, "path": str(path.name)},
            ))
    log.info("Structured: %d chunks", len(chunks))
    return chunks


def _read_rows(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".csv":
        import csv
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


# =====================================================================
# Embedding + indexing
# =====================================================================
class Embedder:
    """sentence-transformers wrapper with a deterministic offline fallback."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self.dim = 384
        self._model = None
        try:
            import os
            if os.getenv("EMBEDDER_OFFLINE", "1") == "1":
                raise RuntimeError("Fast offline embedding mode enabled")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, local_files_only=True)
            self.dim = self._model.get_sentence_embedding_dimension()
            log.info("Embedder: %s (dim=%d)", model_name, self.dim)
        except Exception as exc:
            log.warning("sentence-transformers offline/unavailable (%s) -> fast hashing fallback", exc)

    def encode(self, texts: List[str]):
        import numpy as np
        if self._model is not None:
            return self._model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False)
        # Deterministic hashing embedding: keeps the pipeline runnable
        # offline/air-gapped. Not semantically strong — swap in the real
        # model for production retrieval quality.
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                vecs[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)


class KnowledgeBase:
    """Hybrid (dense + BM25) index over the consolidated corpus."""

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()
        self.chunks: List[Chunk] = []
        self._vectors = None
        self._bm25 = None
        self._qdrant = None

    # ---------- build ----------
    def build(self, chunks: List[Chunk], use_qdrant: bool = True) -> None:
        import numpy as np
        self.chunks = chunks
        if not chunks:
            log.warning("No chunks to index")
            return

        texts = [c.text for c in chunks]
        log.info("Embedding %d chunks ...", len(texts))
        self._vectors = np.asarray(self.embedder.encode(texts), dtype="float32")

        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([t.lower().split() for t in texts])
            log.info("BM25 lexical index ready")
        except ImportError:
            log.warning("rank_bm25 not installed -> dense-only retrieval")

        if use_qdrant:
            self._try_qdrant()
        log.info("Knowledge base ready: %d chunks", len(chunks))

    def _try_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            client = QdrantClient(url=url, timeout=5.0)
            client.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=self._vectors.shape[1],
                                            distance=Distance.COSINE),
            )
            client.upsert(
                collection_name=COLLECTION,
                points=[
                    PointStruct(id=i, vector=self._vectors[i].tolist(),
                                payload=asdict(self.chunks[i]))
                    for i in range(len(self.chunks))
                ],
            )
            self._qdrant = client
            log.info("Qdrant collection '%s' populated at %s", COLLECTION, url)
        except Exception as exc:
            log.info("Qdrant unavailable (%s) -> using in-process index", exc)

    # ---------- retrieve ----------
    def search(self, query: str, top_k: int = 6, alpha: float = 0.6,
               param: Optional[str] = None,
               source_types: Optional[List[str]] = None,
               use_mmr: bool = True) -> List[Dict]:
        """
        Hybrid retrieval.
            alpha = weight on dense similarity (1-alpha on BM25)
            param / source_types = metadata pre-filters
            use_mmr = diversify so we don't return five near-identical chunks
        """
        import numpy as np
        if self._vectors is None or not self.chunks:
            return []

        candidate_idx = [
            i for i, c in enumerate(self.chunks)
            if (param is None or c.param in (None, param))
            and (source_types is None or c.source_type in source_types)
        ]
        if not candidate_idx:
            candidate_idx = list(range(len(self.chunks)))

        qv = np.asarray(self.embedder.encode([query]), dtype="float32")[0]
        dense = self._vectors[candidate_idx] @ qv

        if self._bm25 is not None:
            lex_all = np.asarray(self._bm25.get_scores(query.lower().split()))
            lex = lex_all[candidate_idx]
            lex = (lex - lex.min()) / (np.ptp(lex) or 1.0)
        else:
            lex = np.zeros_like(dense)

        d_norm = (dense - dense.min()) / (np.ptp(dense) or 1.0)
        score = alpha * d_norm + (1 - alpha) * lex

        order = np.argsort(-score)
        picked = (self._mmr(order, candidate_idx, top_k)
                  if use_mmr else order[:top_k].tolist())

        return [{
            "chunk": self.chunks[candidate_idx[i]],
            "score": float(score[i]),
            "dense": float(d_norm[i]),
            "lexical": float(lex[i]),
        } for i in picked]

    def _mmr(self, order, candidate_idx: List[int], top_k: int,
             lambda_: float = 0.72) -> List[int]:
        """Maximal Marginal Relevance for diversity."""
        import numpy as np
        selected: List[int] = []
        pool = order.tolist()[: max(top_k * 4, 20)]
        while pool and len(selected) < top_k:
            if not selected:
                selected.append(pool.pop(0))
                continue
            best, best_score = None, -1e9
            sel_vecs = self._vectors[[candidate_idx[s] for s in selected]]
            for cand in pool:
                v = self._vectors[candidate_idx[cand]]
                redundancy = float(np.max(sel_vecs @ v))
                rank_score = 1.0 - pool.index(cand) / max(len(pool), 1)
                mmr = lambda_ * rank_score - (1 - lambda_) * redundancy
                if mmr > best_score:
                    best, best_score = cand, mmr
            selected.append(best)
            pool.remove(best)
        return selected

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        import numpy as np
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        (path / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], indent=2))
        log.info("Saved knowledge base -> %s", path)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        import numpy as np
        kb = cls()
        kb._vectors = np.load(path / "vectors.npy")
        kb.chunks = [Chunk(**d) for d in json.loads((path / "chunks.json").read_text())]
        try:
            from rank_bm25 import BM25Okapi
            kb._bm25 = BM25Okapi([c.text.lower().split() for c in kb.chunks])
        except ImportError:
            pass
        log.info("Loaded knowledge base (%d chunks) from %s", len(kb.chunks), path)
        return kb


def build_from_data_dir(data_dir: Path = DATA_DIR,
                        use_qdrant: bool = True) -> KnowledgeBase:
    chunks = load_unstructured(data_dir) + load_structured(data_dir)
    kb = KnowledgeBase()
    kb.build(chunks, use_qdrant=use_qdrant)
    return kb


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the BioSCADA knowledge base")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--save", default=str(Path(__file__).resolve().parents[1] / "index"))
    ap.add_argument("--no-qdrant", action="store_true")
    ap.add_argument("--query", default=None, help="test query after building")
    a = ap.parse_args()

    kb = build_from_data_dir(Path(a.data_dir), use_qdrant=not a.no_qdrant)
    if kb.chunks:
        kb.save(Path(a.save))
    if a.query:
        print(f"\nQuery: {a.query}\n" + "-" * 70)
        for r in kb.search(a.query, top_k=5):
            c = r["chunk"]
            print(f"[{r['score']:.3f}] {c.source_type:12s} {c.citation():22s} "
                  f"{c.text[:110].replace(chr(10),' ')}...")
