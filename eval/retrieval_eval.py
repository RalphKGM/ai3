"""Retrieval evaluation on the held-out test set (no LLM needed).

Compares four retriever configurations on the same 50 unseen questions:
  dense      bge-m3 only
  bm25       BM25Okapi only
  hybrid     dense + BM25 fused with RRF(k=60)          (notebook, USE_RERANKER=0)
  hybrid_rr  hybrid top-20 -> bge-reranker-v2-m3 top-5  (notebook default)

Answerable questions -> Hit@1/3/5, section Recall@5, MRR@20, nDCG@5.
Unanswerable questions -> how often the notebook's retrieval abstention gate
(max dense < 0.42 AND max rerank < -2.0) would fire, vs. false gating on
answerable questions.

Also writes data/eval/frozen_top5_cache.jsonl (hybrid_rr top-5) so that the
Qwen and Gemma notebooks generate from identical contexts.

Usage:  .venv/bin/python eval/retrieval_eval.py [--no-rerank] [--no-cache]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from common import EVAL_DIR, OUT_DIR, HybridRetriever, load_test_set, write_jsonl

MIN_DENSE = 0.42
MIN_RERANK = -2.0
K = 5


def rank_metrics(ranked_keys: list[str], gold: set[str]) -> dict:
    """Section-level metrics; duplicates (several parts of one section) count once."""
    seen, dedup = set(), []
    for k in ranked_keys:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    top = ranked_keys[:K]
    first = next((i for i, k in enumerate(dedup, 1) if k in gold), None)
    dcg = sum(1.0 / math.log2(i + 1) for i, k in enumerate(dedup[:K], 1) if k in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), K) + 1))
    return {
        "hit@1": float(bool(ranked_keys[:1]) and ranked_keys[0] in gold),
        "hit@3": float(any(k in gold for k in ranked_keys[:3])),
        "hit@5": float(any(k in gold for k in top)),
        "recall@5": len(gold & set(top)) / len(gold),
        "mrr@20": 1.0 / first if first else 0.0,
        "ndcg@5": dcg / idcg if idcg else 0.0,
        "first_gold_rank": first,
    }


def mean(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true", help="skip the reranker config (much faster on CPU)")
    ap.add_argument("--no-cache", action="store_true", help="do not write frozen_top5_cache.jsonl")
    args = ap.parse_args()

    items = load_test_set()
    t0 = time.perf_counter()
    R = HybridRetriever(load_reranker=not args.no_rerank)
    print(f"Retriever ready on {R.device} in {time.perf_counter() - t0:.1f}s")

    configs = ["dense", "bm25", "hybrid"] + ([] if args.no_rerank else ["hybrid_rr"])
    per_q, cache_rows = [], []

    for n, it in enumerate(items, 1):
        q = it["question"]
        t = time.perf_counter()
        cand = R.candidates(q)
        t_hybrid = time.perf_counter() - t
        rankings = {c: cand[c] for c in ["dense", "bm25", "hybrid"]}
        rr_scores = {}
        t_rr = None
        if not args.no_rerank:
            t = time.perf_counter()
            scored = R.rerank(q, cand["hybrid"])
            t_rr = time.perf_counter() - t
            rankings["hybrid_rr"] = [i for i, _ in scored]
            rr_scores = dict(scored)

        final = rankings["hybrid_rr" if not args.no_rerank else "hybrid"][:K]
        max_dense = max((cand["dense_map"].get(i, 0.0) for i in final), default=0.0)
        max_rr = max((rr_scores[i] for i in final), default=None) if rr_scores else None
        gated = max_dense < MIN_DENSE and (max_rr is None or max_rr < MIN_RERANK)

        row = {
            "id": it["id"],
            "category": it["category"],
            "language": it["language"],
            "is_answerable": it["is_answerable"],
            "gold_section_keys": it["gold_section_keys"],
            "top5_hybrid_rr" if rr_scores else "top5_hybrid": [R.chunks[i]["chunk_id"] for i in final],
            "max_dense_top5": round(max_dense, 4),
            "max_rerank_top5": None if max_rr is None else round(max_rr, 4),
            "gate_would_abstain": gated,
            "latency_hybrid_s": round(t_hybrid, 3),
            "latency_rerank_s": None if t_rr is None else round(t_rr, 3),
        }
        if it["is_answerable"]:
            gold = set(it["gold_section_keys"])
            for c, ranking in rankings.items():
                m = rank_metrics([R.chunks[i]["section_key"] for i in ranking], gold)
                row.update({f"{c}.{k}": v for k, v in m.items()})
        per_q.append(row)

        cache_rows.append({
            **{k: it[k] for k in ["id", "question", "language", "category", "is_answerable",
                                  "ground_truth", "key_facts", "gold_chunk_ids", "gold_section_keys"]},
            "frozen_top5_hits": [R.hit_record(i, cand, rr_scores.get(i)) for i in final],
        })
        print(f"[{n:02d}/{len(items)}] {it['id']} gated={gated} "
              f"max_dense={max_dense:.3f} max_rr={max_rr if max_rr is None else round(max_rr, 2)}")

    ans = [r for r in per_q if r["is_answerable"]]
    unans = [r for r in per_q if not r["is_answerable"]]
    summary = {"n_questions": len(per_q), "n_answerable": len(ans), "n_unanswerable": len(unans),
               "device": R.device, "configs": {}}
    for c in configs:
        summary["configs"][c] = {m: mean(r[f"{c}.{m}"] for r in ans)
                                 for m in ["hit@1", "hit@3", "hit@5", "recall@5", "mrr@20", "ndcg@5"]}
    main_cfg = configs[-1]
    summary["by_category"] = {}
    for cat in sorted({r["category"] for r in ans}):
        rows = [r for r in ans if r["category"] == cat]
        summary["by_category"][cat] = {"n": len(rows), f"{main_cfg}.hit@5": mean(r[f"{main_cfg}.hit@5"] for r in rows),
                                       f"{main_cfg}.mrr@20": mean(r[f"{main_cfg}.mrr@20"] for r in rows)}
    summary["by_language"] = {}
    for lang in sorted({r["language"] for r in ans}):
        rows = [r for r in ans if r["language"] == lang]
        summary["by_language"][lang] = {"n": len(rows), f"{main_cfg}.hit@5": mean(r[f"{main_cfg}.hit@5"] for r in rows),
                                        f"{main_cfg}.mrr@20": mean(r[f"{main_cfg}.mrr@20"] for r in rows)}
    summary["abstention_gate"] = {
        "thresholds": {"min_dense": MIN_DENSE, "min_rerank": MIN_RERANK},
        "unanswerable_gated_rate": mean(r["gate_would_abstain"] for r in unans),
        "answerable_falsely_gated_rate": mean(r["gate_would_abstain"] for r in ans),
        "max_dense_answerable_mean": mean(r["max_dense_top5"] for r in ans),
        "max_dense_unanswerable_mean": mean(r["max_dense_top5"] for r in unans),
    }
    if not args.no_rerank:
        summary["abstention_gate"]["max_rerank_answerable_mean"] = mean(r["max_rerank_top5"] for r in ans)
        summary["abstention_gate"]["max_rerank_unanswerable_mean"] = mean(r["max_rerank_top5"] for r in unans)
    summary["latency_s"] = {
        "hybrid_mean": mean(r["latency_hybrid_s"] for r in per_q),
        "rerank_mean": None if args.no_rerank else mean(r["latency_rerank_s"] for r in per_q),
    }
    summary["misses_at_5"] = [
        {"id": r["id"], "gold": r["gold_section_keys"], "first_gold_rank": r[f"{main_cfg}.first_gold_rank"]}
        for r in ans if not r[f"{main_cfg}.hit@5"]
    ]

    OUT_DIR.mkdir(exist_ok=True)
    write_jsonl(OUT_DIR / "retrieval_eval_per_question.jsonl", per_q)
    (OUT_DIR / "retrieval_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_cache and not args.no_rerank:
        write_jsonl(EVAL_DIR / "frozen_top5_cache.jsonl", cache_rows)
        print(f"Wrote frozen cache -> {EVAL_DIR / 'frozen_top5_cache.jsonl'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
