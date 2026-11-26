"""Leakage check: is the held-out test set really unseen?

Compares every test question against all text used during development:
  * index/retrieval_smoke_test.json      (retrieval tuning queries)
  * data/sft/*/train.jsonl, val.jsonl    (LoRA fine-tuning prompts), if present
  * any extra jsonl passed with --extra  (e.g. Drive copies of the SFT files)

Two kinds of leakage are reported:
  1. Question near-duplicates: word-level Jaccard and character 3-gram
     cosine similarity. Pairs above the thresholds are flagged.
  2. Section overlap with SFT: test gold sections that also appear as the
     target section of an SFT example (reported, not necessarily fatal: the
     model can see a section in training and still be asked a new question
     about it, but the result should be reported separately).

Usage:  python eval/leakage_check.py [--extra path1.jsonl path2.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from common import EVAL_DIR, INDEX_DIR, OUT_DIR, ROOT, load_test_set, normalize, read_jsonl

JACCARD_FLAG = 0.6
CHAR_COS_FLAG = 0.8


def words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize(t)))


def char3(t: str) -> Counter:
    t = normalize(t)
    return Counter(t[i : i + 3] for i in range(len(t) - 2))


def cosine(a: Counter, b: Counter) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def extract_user_question(row: dict) -> str:
    """Pull the user question out of an SFT row (prompt holds system + context + question)."""
    for key in ("question", "query", "user"):
        if isinstance(row.get(key), str):
            return row[key]
    prompt = row.get("prompt", "")
    if isinstance(prompt, list):  # chat-format prompt
        users = [m.get("content", "") for m in prompt if m.get("role") == "user"]
        prompt = users[-1] if users else ""
    m = re.search(r"Question:\s*(.+?)(?:<\|im_end\|>|<end_of_turn>|$)", prompt, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"user\n(.+?)(?:<\|im_end\|>|<end_of_turn>)", prompt, re.S)
    return (m.group(1) if m else prompt[-500:]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", type=Path, default=[])
    args = ap.parse_args()

    test = load_test_set()
    dev: list[tuple[str, str, set]] = []  # (source, question, section_keys)

    smoke = json.loads((INDEX_DIR / "retrieval_smoke_test.json").read_text(encoding="utf-8"))
    dev += [("smoke_test", s["query"], set()) for s in smoke]

    sft_files = sorted((ROOT / "data" / "sft").glob("*/*.jsonl")) + list(args.extra)
    for f in sft_files:
        for row in read_jsonl(f):
            keys = row.get("section_key") or row.get("section_keys") or row.get("gold_section_keys") or []
            keys = {keys} if isinstance(keys, str) else set(keys)
            dev.append((str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f), extract_user_question(row), keys))

    flagged, worst = [], []
    dev_feats = [(src, q, keys, words(q), char3(q)) for src, q, keys in dev]
    for t in test:
        tw, tc = words(t["question"]), char3(t["question"])
        best = (0.0, 0.0, None, None)
        for src, q, _, dw, dc in dev_feats:
            jac = len(tw & dw) / len(tw | dw) if tw | dw else 0.0
            cos = cosine(tc, dc)
            if cos > best[1]:
                best = (jac, cos, src, q)
            if jac >= JACCARD_FLAG or cos >= CHAR_COS_FLAG:
                flagged.append({"test_id": t["id"], "test_q": t["question"], "dev_source": src,
                                "dev_q": q, "jaccard": round(jac, 3), "char3_cos": round(cos, 3)})
        worst.append({"test_id": t["id"], "max_char3_cos": round(best[1], 3), "jaccard_at_max": round(best[0], 3),
                      "closest_dev_source": best[2], "closest_dev_q": best[3]})

    sft_sections = set().union(*(k for s, _, k in dev if s != "smoke_test")) if sft_files else set()
    section_overlap = [
        {"test_id": t["id"], "overlapping_sections": sorted(set(t["gold_section_keys"]) & sft_sections)}
        for t in test if set(t["gold_section_keys"]) & sft_sections
    ]

    report = {
        "dev_sources_checked": sorted({s for s, _, _ in dev}),
        "n_dev_questions": len(dev),
        "sft_files_found": [str(f) for f in sft_files],
        "thresholds": {"jaccard": JACCARD_FLAG, "char3_cosine": CHAR_COS_FLAG},
        "n_flagged_near_duplicates": len(flagged),
        "flagged_near_duplicates": flagged,
        "max_similarity_per_test_question": sorted(worst, key=lambda x: -x["max_char3_cos"]),
        "sft_section_overlap": section_overlap if sft_files else "SFT files not found locally - rerun with data/sft/ present",
        "verdict": "PASS" if not flagged else "REVIEW",
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "leakage_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["dev_sources_checked", "n_dev_questions", "n_flagged_near_duplicates", "verdict"]}, indent=2))
    print("Top-5 closest test/dev pairs:")
    for w in report["max_similarity_per_test_question"][:5]:
        print(f"  {w['test_id']} cos={w['max_char3_cos']} <- {w['closest_dev_source']}: {w['closest_dev_q'][:80]}")
    if not sft_files:
        print("[NOTE] data/sft/ not present locally; SFT prompts not checked yet.")


if __name__ == "__main__":
    main()
