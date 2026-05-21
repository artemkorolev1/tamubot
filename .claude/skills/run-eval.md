---
name: run-eval
description: Guided eval runner — discovers golden sets, confirms settings, runs eval
triggers: ["run eval", "run benchmark", "run evals", "benchmark the pipeline", "benchmark rag", "run chunking eval", "re-run eval", "rerun question", "diff runs"]
---

# Run Eval Skill

Announce: "Using run-eval skill."

## Single runner — `python -m tamubot.evals.run_eval`

The unified runner replaces `run_benchmark.py` and `eval_chunking.py` (both still exist as deprecated shims). Two Makefile entry points:

- `make eval` — retrieval-only (router + retrieval, no generator). Default metrics: `context_precision`, `context_recall`.
- `make eval-gen` — full pipeline with generation. Default metrics: all 4 RAGAS blocks.

Extra flags both targets accept (set as Make vars):

| Var | Effect |
|-----|--------|
| `METRICS=faithfulness,context_recall` | Run subset of RAGAS metrics. Empty = no LLM-judge calls. `METRICS=all` runs every registered block. |
| `IDS=3,7,12` | Re-run only these question ids. The new traces link to the SAME Langfuse dataset run; prior traces get the `superseded` tag. The golden-set `run:<EXP>` column is updated in place. |
| `CAPTURE=1` | Write per-node state sidecar (`tamu_data/evals/state_dumps/<run>.jsonl`) + candidate per-node golden sets (`tamu_data/evals/golden_sets/_node_candidates/<run>/{router,retrieval,generator}.xlsx`). |
| `CHUNKS_COL`, `CHUNK_TAG`, `TOP_K`, `THRESHOLD`, `DESC` | Same meanings as before. |

List available metrics anytime: `python -m tamubot.evals.run_eval --list-metrics`.

## Step 0 — Reuse before re-run (REQUIRED)

Use `scripts/eval_delta.py` to avoid re-running already-evaluated questions:

```bash
# Print IDs missing from any eval_<PREFIX>*.xlsx report
GAP=$(python scripts/eval_delta.py missing tamu_data/evals/golden_sets/<file>.xlsx <EXP_PREFIX>)
```

If `$GAP` is non-empty, run **only those IDs**, using a fresh batch-suffixed `EXP` so each batch's report file survives (`make eval` overwrites `eval_<EXP>.xlsx`):

```bash
make eval EXP=<EXP_PREFIX>_partN IDS=$GAP ...
```

At summary time, combine all batches with one call (no re-run):

```bash
python scripts/eval_delta.py combine <EXP_PREFIX>
```

Use a **new** `EXP_PREFIX` only when code changed and you want a fresh comparison.

**Caveat — `_UPDATED.xlsx` shadowing.** When `<file>_UPDATED.xlsx` exists in `golden_sets/`, `append_run_column` loads from it (FUSE-locked-file fallback) but saves to the original. Edits to the original silently revert on the next run — delete the stale `_UPDATED.xlsx` before editing.

## Step 1 — Discover

Golden sets:
```bash
python -c "
import openpyxl, glob
for f in sorted(glob.glob('tamu_data/evals/golden_sets/*.xlsx')):
    if f.split('/')[-1].startswith('_') or f.split('/')[-1].startswith('~'): continue
    wb = openpyxl.load_workbook(f, read_only=True)
    print(f'{wb.active.max_row - 1:3d}  {f}')
"
```

Chunk collections / tags:
```bash
python -c "
from tamubot.core import config; from pymongo import MongoClient
db = MongoClient(config.MONGODB_URI)[config.MONGODB_DB]
for c in sorted(db.list_collection_names()):
    if not c.startswith('chunks'): continue
    tags = db[c].distinct('chunk_tag')
    print(f'  {c:20s} {db[c].count_documents({}):4d} docs  tags: {\", \".join(tags) or \"(none)\"}')
"
```

Langfuse datasets:
```bash
python -c "
from langfuse import Langfuse
lf = Langfuse()
for ds in lf.api.datasets.list(page=1, limit=50).data:
    items = lf.api.dataset_items.list(dataset_name=ds.name, page=1, limit=1)
    print(f'  {ds.name:40s}  {items.meta.total_items:3d} items')
"
```

## Step 2 — Ask the user (always, no defaults)

Single AskUserQuestion call with 4 choices — never assume defaults:

1. **Mode** — `make eval` (retrieval only) or `make eval-gen` (with generation).
2. **Golden set** — list ALL discovered golden sets as options.
3. **Collection + chunk tag** — list discovered combinations.
4. **Metric selection** — options: `default for mode`, `all`, or a specific subset (e.g. `context_recall` only for fast iteration).

Then a second AskUserQuestion:

5. **Capture state?** — yes (writes sidecar + per-node candidates) or no.
6. **Re-run subset?** — if user is iterating, run Step 0 first; propose `IDS=$GAP`, never `IDS=<all>` when partial results exist.

## Naming conventions (required)

Every run needs a name that identifies **what is being tested**, plus a description that gives future-you the context to interpret the result. The Langfuse public API does NOT support renaming or updating descriptions after the fact via standard endpoints — get it right the first time.

### Run name (`EXP=`)

Format: `<change>_<YYYYMMDD>[_v<N>]` — lowercase snake_case.

- `<change>` describes **what's being tested** (the experiment), not what subset of rows ran. Good: `discovery_guard`, `subq_fanout`, `baseline_pre_fanout`, `chunk_400t`. Bad: `rest17`, `partial5`, `q3517_v4`, `quick_test`.
- `_v<N>` only when re-running the same change on the same day (e.g., `discovery_guard_20260521_v2`).
- If iterating with `IDS=…` against an existing run, KEEP the original name — the runner appends to the same run.

### Description (`DESC=`, REQUIRED)

Two short lines (passed through `--description`). Include:

1. **Branch + short commit SHA** of the code under test.
2. **What changed** since the last comparison run (and its name) — or `baseline` if first.
3. **Coverage** if not the full dataset (e.g., `IDS=2,8,9` or `partial 3/20`).

Template:

```
branch=<branch>@<sha7> | <change-summary> vs <prior_run_name>
coverage=<n>/<total> from <golden-set-stem>
```

Example:

```
branch=router-subqueries-fanout@30cf2b2 | title-recovery discovery-pattern guard vs router_subqueries_30cf2b2_full
coverage=20/20 from ragas_20260519_curated20_v2
```

### Re-runs with `IDS=` keep the existing description

The runner sets the description on the parent run during the initial full execution. Subsequent `IDS=` reruns inherit it. So: when starting a new experimental run that you expect to iterate on with `IDS=`, set the description on the first call.

## Step 3 — Confirm

Print the resolved command before invoking. Wait for explicit "yes / go / run" before executing — gate is non-skippable per `feedback_eval_confirm_all_settings`.

```
Plan to run:
  GOLDEN=tamu_data/evals/golden_sets/<file>.xlsx
  EXP=<experiment>
  Mode=<eval | eval-gen>
  METRICS=<…>
  IDS=<… or empty>
  CAPTURE=<1 or empty>
  CHUNKS_COL=<…>  CHUNK_TAG=<…>
  DESC=<…>
```

Invoke `task-budget` skill before running if a metric set will incur LLM-judge calls.

## Step 4 — Run

```bash
make eval     GOLDEN=… EXP=… METRICS=… [IDS=… CAPTURE=1 …]
make eval-gen GOLDEN=… EXP=… METRICS=… [IDS=… CAPTURE=1 …]
```

Stream output live.

## Step 5 — Report

Print: router accuracy from the per-query tab, error count, Langfuse run URL, Excel report path. If `CAPTURE=1`: also report sidecar path + node-candidates dir.

## Comparing two runs

```bash
make diff-runs \
    GOLDEN=tamu_data/evals/golden_sets/<file>.xlsx \
    LEFT=run:<exp_a> RIGHT=run:<exp_b> \
    OUTPUT=tamu_data/evals/reports/diff_<a>_vs_<b>.xlsx
```

Writes a side-by-side xlsx with `id, question, left_answer, right_answer, changed`. Optional `METRIC=<name>` adds numeric delta + green/red fills when matching `run:<exp>:<metric>` score columns are present.
