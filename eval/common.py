"""Shared helpers for Botobot test-set evaluation.

The retriever here mirrors `retrieve()` in 03/04_Run_*_RAG.ipynb (Cell 6):
bge-m3 CLS-pooled dense search (FAISS inner product) + BM25Okapi, fused with
RRF (k=60) over the top-20 candidates of each, then an optional
bge-reranker-v2-m3 stage-2 rerank down to the top-k.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index"
EVAL_DIR = ROOT / "data" / "eval"
OUT_DIR = ROOT / "eval_outputs"

EMBED_MODEL_ID = "BAAI/bge-m3"
RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"

LEGAL_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize_legal(text: str) -> list[str]:
    return LEGAL_TOKEN_RE.findall(text.lower())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_chunks() -> list[dict]:
    return read_jsonl(INDEX_DIR / "chunks_index.jsonl")


def load_test_set(path: Path | None = None) -> list[dict]:
    """Load the held-out questions and attach gold section keys.

    Gold is scored at section level: a long section split into `_partN` chunks
    counts as a hit if any of its parts is retrieved.
    """
    chunks = {c["chunk_id"]: c for c in load_chunks()}
    items = read_jsonl(path or EVAL_DIR / "test_50_questions.jsonl")
    for it in items:
        missing = [c for c in it["gold_chunk_ids"] if c not in chunks]
        if missing:
            raise ValueError(f"{it['id']}: unknown gold chunk ids {missing}")
        it["gold_section_keys"] = sorted({chunks[c]["section_key"] for c in it["gold_chunk_ids"]})
    return items


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


class HybridRetriever:
    """Dense + BM25 + RRF + optional reranker, identical to the notebook."""

    def __init__(self, device: str | None = None, load_reranker: bool = True):
        import faiss
        import torch
        from rank_bm25 import BM25Okapi
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.chunks = load_chunks()
        emb = np.load(INDEX_DIR / "bge_m3_embeddings.npy").astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.bm25 = BM25Okapi(json.loads((INDEX_DIR / "bm25_tokens.json").read_text(encoding="utf-8")))
        assert self.index.ntotal == len(self.chunks), "index/chunk count mismatch"

        self.embed_tok = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
        self.embed_model = AutoModel.from_pretrained(EMBED_MODEL_ID).to(device).eval()

        self.rerank_tok = self.rerank_model = None
        if load_reranker:
            self.rerank_tok = AutoTokenizer.from_pretrained(RERANK_MODEL_ID)
            self.rerank_model = (
                AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_ID).to(device).eval()
            )

    def encode_query(self, query: str) -> np.ndarray:
        torch = self.torch
        enc = self.embed_tok([query], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            cls = self.embed_model(**enc).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
        return cls.float().cpu().numpy()

    def candidates(self, query: str, candidate_k: int = 20, rrf_k: int = 60) -> dict:
        """Return dense, BM25 and fused rankings (lists of chunk indices) plus raw scores."""
        dense_scores, dense_idxs = self.index.search(self.encode_query(query), candidate_k)
        dense_rank = [int(i) for i in dense_idxs[0] if i >= 0]
        dense_map = {int(i): float(s) for i, s in zip(dense_idxs[0], dense_scores[0]) if i >= 0}

        bm25_scores = self.bm25.get_scores(tokenize_legal(query))
        bm25_rank = [int(i) for i in np.argsort(bm25_scores)[::-1][:candidate_k] if bm25_scores[int(i)] > 0]

        rrf: dict[int, float] = {}
        for rank, idx in enumerate(dense_rank, 1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, idx in enumerate(bm25_rank, 1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        fused = [i for i, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:candidate_k]]
        return {
            "dense": dense_rank,
            "bm25": bm25_rank,
            "hybrid": fused,
            "dense_map": dense_map,
            "bm25_scores": bm25_scores,
            "rrf": rrf,
        }

    def rerank(self, query: str, idxs: list[int], batch_size: int = 4) -> list[tuple[int, float]]:
        torch = self.torch
        scored = []
        for s in range(0, len(idxs), batch_size):
            batch = idxs[s : s + batch_size]
            pairs = [[query, self.chunks[i]["text"]] for i in batch]
            enc = self.rerank_tok(pairs, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.rerank_model(**enc).logits.view(-1).float().cpu().tolist()
            scored.extend(zip(batch, logits))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def hit_record(self, idx: int, cand: dict, rerank_score: float | None = None) -> dict:
        rec = dict(self.chunks[idx])
        rec["dense_score"] = cand["dense_map"].get(idx, 0.0)
        rec["bm25_score"] = float(cand["bm25_scores"][idx])
        rec["rrf_score"] = float(cand["rrf"].get(idx, 0.0))
        rec["rerank_score"] = rerank_score
        return rec
