"""Score generated answers from the RAG notebooks against the held-out test set.

Input: one or more `<model>_<precision>_preds.jsonl` files written by Cell 9 of
03/04_Run_*_RAG.ipynb (fields: id, answer, abstained, retrieved_chunk_ids,
retrieved_contexts, latency_s, ...).

Metrics (all deterministic, no LLM judge):
  answerable questions
    key_fact_recall   share of required facts present in the answer
    strict_accuracy   all required facts present and not abstained
    token_f1          SQuAD-style token F1 vs. the reference answer
    citation_correct  answer names the gold Section/Article number, or cites
                      a [n] marker whose chunk is a gold section
    false_abstention  abstained although the answer was in the corpus
  unanswerable questions
    correct_abstention  abstained (the desired behaviour)
    hallucination       answered instead of abstaining
  grounding (faithfulness proxies, non-abstained answers)
    number_grounding    share of numbers/times in the answer that also appear
                        in the retrieved context (ungrounded number = red flag)
    sentence_support    share of answer sentences whose content words are
                        >=50% covered by the retrieved context
  retrieval -> generation link
    strict accuracy split by whether a gold section was in the top-5 context
  latency          mean / p50 / p95 seconds, tokens

Outputs per preds file in eval_outputs/:
  <stem>_scored.jsonl, <stem>_score_summary.json,
  <stem>_human_review.csv  (sheet for manual 1-5 relevance/faithfulness rating)
and a side-by-side eval_outputs/model_comparison.md when several files are given.

Usage:  python eval/score_predictions.py eval_outputs/qwen35_8bit_preds.jsonl [more...]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from common import OUT_DIR, load_chunks, load_test_set, normalize, read_jsonl, write_jsonl

ABSTAIN_MARKERS = [
    "cannot answer this based on the provided",
    "not covered by the provided",
    "outside the documented scope",
    "hindi saklaw ng",
    "wala sa ibinigay na",
    "not found in the provided",
    "does not contain",
    "do not contain",
    "not mentioned in the provided",
    "no information",
    "walang impormasyon",
    "hindi ko masasagot",
]

STOP = set("""a an the of to in on for and or by with is are be was were it its this that as at from
shall may can not no under per any all their his her they them who what when where which how
ng sa ang mga na at ay para kung ko mo po ba si ni nang""".split())

NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")


def is_abstention(text: str) -> bool:
    low = normalize(text)
    return any(m in low for m in ABSTAIN_MARKERS)


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize(text)) if t not in STOP]


def token_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    common = {}
    for t in p:
        if t in r:
            common[t] = min(p.count(t), r.count(t))
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def fact_norm(text: str) -> str:
    """Normalize for fact matching: '(30)-meter' == '30 meter', '12,000' == '12000'."""
    t = normalize(text).replace(",", "")
    t = re.sub(r"[()\-–\"]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def key_fact_recall(answer: str, key_facts: list[list[str]]) -> float:
    if not key_facts:
        return 1.0
    a = fact_norm(answer)
    hits = sum(any(fact_norm(alt) in a for alt in group) for group in key_facts)
    return hits / len(key_facts)


_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_NUMWORD_RE = re.compile(r"\b(" + "|".join(_TENS) + r")(?:[- ](" + "|".join(_UNITS[1:10]) + r"))?\b|\b(" + "|".join(_UNITS) + r")\b")


def words_to_digits(text: str) -> str:
    """'ninety days' -> 'ninety 90 days'; keeps the word so phrase matching still works."""
    def rep(m):
        if m.group(1):
            n = _TENS[m.group(1)] + (_UNITS.index(m.group(2)) if m.group(2) else 0)
        else:
            n = _UNITS.index(m.group(3))
        return f"{m.group(0)} {n}"
    return _NUMWORD_RE.sub(rep, text)


def section_numbers(section_key: str) -> list[str]:
    """'comres_11076::ARTICLE_VI::SEC_36' -> ['36']; 'const_1987::ARTICLE VI::Sec 3' -> ['3']."""
    tail = section_key.split("::")[-1]
    return re.findall(r"\d+", tail)


def citation_correct(answer: str, gold_keys: list[str], retrieved_ids: list[str], chunk_sec: dict) -> bool:
    a = normalize(answer)
    for i, cid in enumerate(retrieved_ids, 1):
        if f"[{i}]" in a and chunk_sec.get(cid) in gold_keys:
            return True
    for key in gold_keys:
        for num in section_numbers(key):
            if re.search(rf"\b(sec\.?|section|seksyon)\s*{num}\b", a):
                return True
    return False


def number_grounding(answer: str, contexts: list[str]) -> float | None:
    ctx = words_to_digits(normalize(" ".join(contexts))).replace(",", "")
    nums = {n.replace(",", "") for n in NUM_RE.findall(normalize(answer))}
    # ignore list markers / chunk markers like [1]..[5] and the resolution numbers themselves
    nums -= {"1", "2", "3", "4", "5", "11076", "11086", "1987"}
    if not nums:
        return None
    return sum(n in ctx for n in nums) / len(nums)


def sentence_support(answer: str, contexts: list[str]) -> float | None:
    ctx = set(tokens(" ".join(contexts)))
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(tokens(s)) >= 4]
    if not sents:
        return None
    supported = sum(len([t for t in tokens(s) if t in ctx]) / len(tokens(s)) >= 0.5 for s in sents)
    return supported / len(sents)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def score_file(path: Path, test: dict, chunk_sec: dict) -> dict:
    preds = read_jsonl(path)
    rows = []
    for p in preds:
        t = test.get(p["id"])
        if t is None:
            print(f"[WARN] {p['id']} not in test set; skipped")
            continue
        ans = p.get("answer", "")
        abst = bool(p.get("abstained")) or is_abstention(ans)
        gold_in_ctx = bool(set(t["gold_section_keys"]) & {chunk_sec.get(c) for c in p.get("retrieved_chunk_ids", [])})
        r = {"id": p["id"], "category": t["category"], "language": t["language"],
             "is_answerable": t["is_answerable"], "abstained": abst,
             "gold_in_top5": gold_in_ctx if t["is_answerable"] else None,
             "latency_s": p.get("latency_s"), "gen_tokens": p.get("gen_tokens")}
        if t["is_answerable"]:
            kfr = 0.0 if abst else key_fact_recall(ans, t["key_facts"])
            r.update({
                "key_fact_recall": kfr,
                "strict_correct": (not abst) and kfr == 1.0,
                "token_f1": 0.0 if abst else round(token_f1(ans, t["ground_truth"]), 4),
                "citation_correct": (not abst) and citation_correct(ans, t["gold_section_keys"], p.get("retrieved_chunk_ids", []), chunk_sec),
            })
        else:
            r["correct_abstention"] = abst
        if not abst:
            r["number_grounding"] = number_grounding(ans, p.get("retrieved_contexts", []))
            r["sentence_support"] = sentence_support(ans, p.get("retrieved_contexts", []))
        rows.append(r)

    ans = [r for r in rows if r["is_answerable"]]
    un = [r for r in rows if not r["is_answerable"]]
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    first = preds[0] if preds else {}
    summary = {
        "preds_file": str(path),
        "model_id": first.get("model_id"), "precision": first.get("precision"),
        "use_lora": first.get("use_lora"), "gpu_sku": first.get("gpu_sku"),
        "n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(un),
        "answerable": {
            "strict_accuracy": mean(r["strict_correct"] for r in ans),
            "key_fact_recall": mean(r["key_fact_recall"] for r in ans),
            "token_f1": mean(r["token_f1"] for r in ans),
            "citation_accuracy": mean(r["citation_correct"] for r in ans),
            "false_abstention_rate": mean(r["abstained"] for r in ans),
        },
        "unanswerable": {
            "correct_abstention_rate": mean(r["correct_abstention"] for r in un),
            "hallucination_rate": mean(not r["correct_abstention"] for r in un),
        },
        "grounding": {
            "number_grounding": mean(r.get("number_grounding") for r in rows),
            "sentence_support": mean(r.get("sentence_support") for r in rows),
        },
        "retrieval_effect": {
            "accuracy_when_gold_in_top5": mean(r["strict_correct"] for r in ans if r["gold_in_top5"]),
            "accuracy_when_gold_missing": mean(r["strict_correct"] for r in ans if not r["gold_in_top5"]),
            "n_gold_in_top5": sum(bool(r["gold_in_top5"]) for r in ans),
        },
        "by_language": {
            lang: {"n": len(g), "strict_accuracy": mean(r["strict_correct"] for r in g)}
            for lang in sorted({r["language"] for r in ans}) for g in [[r for r in ans if r["language"] == lang]]
        },
        "by_category": {
            cat: {"n": len(g), "strict_accuracy": mean(r["strict_correct"] for r in g),
                  "key_fact_recall": mean(r["key_fact_recall"] for r in g)}
            for cat in sorted({r["category"] for r in ans}) for g in [[r for r in ans if r["category"] == cat]]
        },
        "latency_s": {
            "mean": mean(lat),
            "p50": round(float(np.percentile(lat, 50)), 3) if lat else None,
            "p95": round(float(np.percentile(lat, 95)), 3) if lat else None,
        },
        "failures": {
            "wrong_or_incomplete": [r["id"] for r in ans if not r["strict_correct"] and not r["abstained"]],
            "false_abstentions": [r["id"] for r in ans if r["abstained"]],
            "hallucinated_on_unanswerable": [r["id"] for r in un if not r["correct_abstention"]],
            "ungrounded_numbers": [r["id"] for r in rows if (r.get("number_grounding") or 1.0) < 1.0],
        },
    }

    stem = path.stem
    write_jsonl(OUT_DIR / f"{stem}_scored.jsonl", rows)
    (OUT_DIR / f"{stem}_score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT_DIR / f"{stem}_human_review.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "reference_answer", "model_answer", "auto_strict_correct",
                    "relevance_1to5", "faithfulness_1to5", "fluency_1to5", "reviewer", "notes"])
        by_id = {p["id"]: p for p in preds}
        for r in rows:
            t = test[r["id"]]
            w.writerow([r["id"], t["question"], t["ground_truth"], by_id[r["id"]].get("answer", ""),
                        r.get("strict_correct", r.get("correct_abstention")), "", "", "", "", ""])
    return summary


def comparison_table(summaries: list[dict]) -> str:
    cols = [
        ("Strict accuracy", lambda s: s["answerable"]["strict_accuracy"]),
        ("Key-fact recall", lambda s: s["answerable"]["key_fact_recall"]),
        ("Token F1", lambda s: s["answerable"]["token_f1"]),
        ("Citation acc.", lambda s: s["answerable"]["citation_accuracy"]),
        ("False abstention", lambda s: s["answerable"]["false_abstention_rate"]),
        ("Correct abstention", lambda s: s["unanswerable"]["correct_abstention_rate"]),
        ("Number grounding", lambda s: s["grounding"]["number_grounding"]),
        ("Mean latency (s)", lambda s: s["latency_s"]["mean"]),
        ("p95 latency (s)", lambda s: s["latency_s"]["p95"]),
    ]
    head = "| Run | " + " | ".join(c for c, _ in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    lines = [head, sep]
    for s in summaries:
        name = f"{(s['model_id'] or '?').split('/')[-1]} {s['precision'] or ''} {'+LoRA' if s['use_lora'] else 'base'}"
        lines.append(f"| {name} | " + " | ".join("" if f(s) is None else str(f(s)) for _, f in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", type=Path)
    args = ap.parse_args()
    test = {t["id"]: t for t in load_test_set()}
    chunk_sec = {c["chunk_id"]: c["section_key"] for c in load_chunks()}
    OUT_DIR.mkdir(exist_ok=True)
    summaries = [score_file(p, test, chunk_sec) for p in args.preds]
    for s in summaries:
        print(json.dumps(s, indent=2))
    if len(summaries) > 1:
        table = comparison_table(summaries)
        (OUT_DIR / "model_comparison.md").write_text("# Held-out test set: model comparison\n\n" + table + "\n", encoding="utf-8")
        print(table)


if __name__ == "__main__":
    main()
