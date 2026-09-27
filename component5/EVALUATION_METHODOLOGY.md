# Evaluation methodology and limits

`component5/run_eval.py` produces the measured report at `component5/out/benchmark_report.md` and its per-turn data in `turn_log.jsonl`.

The harness uses seven labelled sessions and 24 turns over a hand-authored local fixture corpus. It compares always-retrieve, controller plus decomposer, and the full session-aware pipeline. The retrieval ablations compare hybrid, dense-only, and sparse-only search with reranking on and off.

The default models are a token hashing embedder and token-overlap reranker. Reported latency excludes model loading and network inference. Citation support is exact source text matching for this fixture, not a general entailment metric. Use the intended corpus, hold-out labels, real model backends, and human-reviewed claim support before making production claims.
