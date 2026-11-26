# Held-out test set evaluation (Botobot / Group 6)

Covers rubric items **Model Testing (Unseen Data)**, **Correctness of Training** and
**Results and Analysis**, and notebook Section 5.

## Test set: `data/eval/test_50_questions.jsonl`

50 questions written after the index and SFT data were frozen. None were used for
retrieval tuning, prompt design or fine-tuning (see the leakage check below).

| Split | n | What it tests |
|---|---:|---|
| Answerable, Res. 11086 (campaign rules) | 13 | dates, sizes, peso limits, procedures |
| Answerable, Res. 11076 (election-day ops) | 20 | voting hours, EB rules, PPP/PDL voting, prohibitions |
| Answerable, 1987 Constitution | 6 | COMELEC, Senate/House/President qualifications, suffrage (t37 spans Art. VII Sec 2 + Sec 4) |
| Answerable, multi-hop across documents | 1 | t40: Constitution Art. IX-C Sec 8 + Res. 11076 SEC. 14 |
| Unanswerable (abstention traps) | 10 | 2028/2026 dates, results, registration, penalty years, off-domain |
| *Language* | 39 EN / 11 Filipino-Taglish | |

Each row has these fields:
- `question`, `language`, `category`, `is_answerable`
- `ground_truth`: a reference answer written from the source text
- `key_facts`: a list of fact groups; each group lists accepted surface forms. Every group must appear for an answer to be strictly correct.
- `gold_chunk_ids`: the chunk(s) holding the answer. Scoring is done at section level, so any `_partN` of the gold section counts.

Every key fact was checked against its gold chunk text, and every unanswerable
question was checked against the whole corpus. One candidate question (Commissioner
salary) turned out to be answerable from Art. XVIII Sec 17 and was replaced.

## How to run (Colab)

1. Upload these to `MyDrive/AI3_Botobot/`, next to the existing `data/index/` and `data/sft/`:
   - the `eval/` folder
   - `data/eval/test_50_questions.jsonl`
2. **`05_Evaluate_Test_Set.ipynb`, Part A (Cells 1–6)** on a GPU runtime:
   - leakage check against the smoke-test queries and every `data/sft/*/*.jsonl` prompt
   - retrieval evaluation of 4 retriever configurations
   - writes `data/eval/frozen_top5_cache.jsonl`
3. **Notebooks 03 / 04, Cells 1–9.** Cell 8 loads the frozen cache, so every model, precision and LoRA variant answers from the same top-5 contexts. Any difference between runs is then due to generation alone. Rename each `*_preds.jsonl` after a run (e.g. `qwen35_8bit_lora_preds.jsonl`), because base and LoRA runs at the same precision otherwise overwrite each other.
4. **`05`, Part B (Cells 7–9)** (re-run Cells 1–2 first in a new session):
   - scores every `eval_outputs/*_preds.jsonl`
   - builds the model comparison table and charts
   - lists failure cases next to the context each answer was given

| Script | Output (in `eval_outputs/`) |
|---|---|
| `leakage_check.py` | `leakage_report.json` |
| `retrieval_eval.py` | `retrieval_eval_summary.json`, `retrieval_eval_per_question.jsonl` (+ frozen cache) |
| `score_predictions.py` | `<run>_scored.jsonl`, `<run>_score_summary.json`, `<run>_human_review.csv`, `model_comparison.md` |
| notebook 05 | `generation_metrics_table.csv`, `figures/retrieval_eval.png`, `figures/generation_eval.png` |

## Metrics

**Retrieval** (section level, answerable only): Hit@1/3/5, Recall@5, MRR@20, nDCG@5,
for dense-only, BM25-only, hybrid RRF and hybrid + reranker. For the abstention gate,
it reports how often the notebook's `min_dense=0.42 / min_rerank=-2.0` gate would fire
on unanswerable questions and how often it would fire wrongly on answerable ones.

**Generation** (deterministic, no LLM judge):
- *strict accuracy*: every key fact is present and the model did not abstain. Also reported: key-fact recall and token F1 against the reference.
- *citation accuracy*: the answer names the gold section number, or its `[n]` marker points at a gold chunk.
- *false abstention* (answerable) and *correct abstention / hallucination rate* (unanswerable).
- faithfulness proxies:
  - *number grounding*: every number in the answer must appear in the retrieved context. Spelled-out numbers like "ninety" are converted to digits first.
  - *sentence support*: at least 50% of each sentence's content words must appear in the context.
- *retrieval effect*: accuracy when the gold section was in the top 5 vs. when it was missed.
- latency mean/p50/p95.

`*_human_review.csv` is a blank rating sheet (relevance / faithfulness / fluency, 1–5)
for the user evaluation the spec asks for. Each member can rate a share of the rows.

**Scorer validation.** The scorer was run on two synthetic prediction files:
- answers equal to the references: 100% strict accuracy, 100% citation accuracy, 100% correct abstention
- a dummy model that always hallucinates: 0% on all three

Known proxy limit: ordinal dates ("thirtieth day of June" vs "30 June") count as
ungrounded numbers, so the reference answers' number grounding tops out at 0.96, not 1.0.

## Known limitations
- Key-fact matching is string based, so an answer that is correct but paraphrases a fact unusually can be marked wrong. The human-review sheet is the check for this.
- 50 questions gives wide confidence intervals: one question is 2.5 points of answerable accuracy.
- On Colab the leakage check picks up `data/sft/*/*.jsonl` automatically. Run locally, it only covers the smoke-test queries unless the SFT files are copied into `data/sft/`.
