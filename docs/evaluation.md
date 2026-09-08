# KrishiMitra AI — Evaluation

## Philosophy

The **primary** harness (`app/evaluation/evaluator.py` + `scripts/evaluate.py`)
is deterministic and needs no API key: it checks exact, verifiable
properties (was a claim actually cited? did the guardrail actually refuse?
did the eligibility engine produce a defensible verdict?) rather than
subjective judgments. This is what CI and the demo rely on.

**DeepEval** (`app/evaluation/deepeval_runner.py`) is an *optional*,
additive LLM-as-judge layer for qualitative signal (Faithfulness, Answer
Relevancy) on top of that — disabled by default (`DEEPEVAL_ENABLED=false`),
since it costs an API call per case and is a judgment call rather than an
exact check. Enable with `--deepeval` on `scripts/evaluate.py`.

## Golden dataset

`app/evaluation/test_cases.json` — 33 cases across 10 categories: direct
factual, eligibility, missing-information, out-of-domain, hallucination
traps, prompt injection, objection handling, multilingual, document-upload,
and ambiguous queries. Each case declares `must_refuse` and
`expect_relevant` so the harness can score behaviour, not just check the
answer "looks reasonable".

## Metrics

| Metric | What it measures |
|---|---|
| Retrieval Precision | Of cases with any retrieved chunk, % that passed the evidence-confidence threshold |
| Retrieval Recall | Of cases expected to be answerable, % where at least one chunk was retrieved |
| Groundedness | Of non-refused answers, % where every surviving claim is cited (true by construction post-guard, but tracked to catch regressions) |
| Citation Accuracy | Of non-refused answers, % with at least one citation pointing at real evidence |
| Relevance Accuracy | Of answerable cases, % where `relevant` matched the expected value |
| Hallucination Rate | Of non-refused answers, % where the guard had to delete ≥1 unsupported claim |
| Refusal Accuracy | Of cases that must refuse (off-domain, injection, hallucination traps), % that actually refused |
| Eligibility Accuracy | Of eligibility-category cases, % where the verdict was defensible (never "likely_eligible" without a matched, cited criterion) |

## Running it

```bash
python scripts/evaluate.py                 # full run, deterministic metrics only
python scripts/evaluate.py --limit 10       # quick smoke run
python scripts/evaluate.py --deepeval       # + optional LLM-as-judge metrics
python scripts/evaluate.py --json out.json  # save full per-case results
```

## Known calibration note

The evidence-confidence threshold (`RETRIEVAL_SCORE_THRESHOLD`, default
0.32) gates on raw vector cosine similarity of the top chunk. Sentence-
embedding models tend to report a "high floor" of similarity between any two
same-domain sentences (e.g. two different farmer-scheme sentences can score
0.35–0.5 even when topically unrelated), especially against a small corpus.
This is a property of the embedding model, not the pipeline — the system's
actual hard guarantee is structural, not threshold-based: **no scheme card
or claim is ever shown without a real, verifiable citation**, which holds
regardless of where the threshold is tuned. With the full crawled corpus
(dozens of schemes rather than one), the relative separation between
genuinely relevant and irrelevant chunks is much clearer than in a 2–3
document unit test.
