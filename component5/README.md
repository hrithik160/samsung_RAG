# Component 5 - Telemetry + Eval (Phase 5)

Runs labelled multi-turn sessions through three system variants, logs one JSON
record per turn, and summarises with Pandas + Pillow.

```bash
python component5/run_eval.py          # -> component5/out/{turn_log.jsonl, summary.csv, ablations, gates, chart, report}
python component5/test_component5.py   # 10 tests
```

The repository root's `pylock.toml` includes the pinned offline evaluation dependencies. Prefer `./run.ps1 -Eval` for a fresh Windows setup.

## The three variants

| Variant | Pipeline | Session memory |
|---|---|---|
| **A_always_retrieve** | raw utterance -> Component 3 | none |
| **B_controller_only** | C1 gate -> C2 decompose -> C3 | none (C1 NO_RETRIEVAL = no answer) |
| **C_full_pipeline** | C1 -> C2 -> C3 -> C4 | versioned session cache |

All three share the same index, embedder and reranker, so differences come from
the *pipeline*, not the retrieval models.

## Metrics

| Metric | Definition |
|---|---|
| retrieval quality | recall@3 and MRR over turns that need evidence (`should_retrieve`). A turn where the system didn't retrieve scores 0. |
| latency | wall-clock ms for the whole turn (controller + decompose + retrieve + synth) |
| retrievals | number of Component 3 calls |
| cache reuse | share of *cosmetic* turns answered from the previous answer with no retrieval |
| refinement accuracy | over `refine` + `cosmetic` turns. refine = retrieved AND all gold docs in top-3. cosmetic = did not retrieve AND previous citations survive in the answer |
| decision accuracy | `retrieved == should_retrieve` over all turns |

Labels (`eval_data.py`) are written from user intent, not from component output.
Add sessions to `SESSIONS` to grow the set; `test_dataset_integrity` guards the schema.

## Reading the output

`turn_log.jsonl` has one row per (variant, session, turn) with decision, latency,
retrieved docs, gold docs, and correctness flags - use it to see *which* turns fail:

```python
import pandas as pd
df = pd.read_json("component5/out/turn_log.jsonl", lines=True)
df[~df.task_correct & (df.variant == "C_full_pipeline")]
```

## Caveats

- Latency is measured on the offline stand-ins (hashing embedder, token-overlap
  reranker, rule-based decomposer/synthesizer). Treat it as *relative overhead*, not
  production latency. Re-run with real BGE + LLM backends for real numbers.
- 7 sessions / 24 turns is a smoke-test-sized set, not a benchmark.
- `out/benchmark_report.md` and `out/acceptance_gates.csv` are generated fixture results; G1 stays pending until a clean-machine replay.
