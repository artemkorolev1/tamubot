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
6. **Re-run subset?** — if user is iterating after a prior run, ask for `IDS=…` (use the same EXP name to overwrite the dataset run).

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
