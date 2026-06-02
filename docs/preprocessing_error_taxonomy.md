# Preprocessing Error Taxonomy — v6b judge contract

**Version: `1.0.0`** · frozen. Every judge verdict emits error tags drawn **only** from
the closed enum below. Changing this file changes the contract: bump the version, and
note that cross-run "new vs cleared error type" diffs are only comparable **within the
same taxonomy version**. The version string is recorded in each run's `manifest.jsonl`.

## Why this exists

The judge (Inspect model-graded scorer) compares an `original/` view (faithful extracted
text) against the `processed/` view (what RAG actually sees after boilerplate/dedup/chunk)
and, for fidelity, against the **source PDF render**. Each finding must:

1. name a **type** from the closed enum,
2. carry a **severity** (`blocker` > `major` > `minor`),
3. attribute to an **owning stage** + **owning code file** as a *ranked hypothesis with
   confidence* (the judge sees endpoints, not intermediate artifacts — never force a
   single owner),
4. cite **evidence** (a quoted span / chunk index / PDF page).

Stage attribution is the point: a finding tells us *which algorithm to change*. The judge
**defers to the deterministic checks** (`silver_chunk_checks`, `silver_tag_checks`) for
rates and counts — it judges *individual decisions*, not corpus rates.

## Dimensions

| Dimension | Question the judge answers | Reference |
|---|---|---|
| `boilerplate` | Was shared/legal/admin boilerplate correctly hidden, without hiding real content? | original ↔ processed + `.why.json` |
| `dedup` | Were duplicates correctly collapsed to the right canonical, without false merges? | original ↔ processed + `.why.json` |
| `chunking` | Are chunk boundaries semantically sensible (no orphaned headers, oversize, fragments, mid-sentence cuts)? | processed boundaries |
| `fidelity` | Did real content (tables, images, headers, prose) survive extraction without loss or hallucination? | **source PDF** ↔ original/processed |

## Closed error-type enum

### Dimension: `boilerplate` — owner stage `silver_tag`
| Type | Severity | Meaning | Owning code file(s) |
|---|---|---|---|
| `BP_MISSED` | major | Real boilerplate left in the processed view (not flagged) | `pipeline_v6b/util/tagging.py`, `util/boilerplate_clustering.py` |
| `BP_OVERREACH` | blocker | Real, syllabus-specific content wrongly flagged as boilerplate and hidden | `pipeline_v6b/util/tagging.py`, `util/boilerplate_clustering.py` |
| `BP_WRONG_CLUSTER` | minor | Flagged boilerplate, but attributed to a clearly wrong reference cluster | `pipeline_v6b/util/text_normalize.py` (ReferenceIndex) |

### Dimension: `dedup` — owner stage `silver_tag`
| Type | Severity | Meaning | Owning code file(s) |
|---|---|---|---|
| `DUP_MISSED` | minor | Genuine duplicate left un-collapsed (both copies visible) | `pipeline_v6b/util/tagging.py`, `util/signature_index.py` |
| `DUP_FALSE` | blocker | Distinct content wrongly flagged as a duplicate and hidden | `pipeline_v6b/util/tagging.py` (thresholds 0.92/0.95) |
| `DUP_WRONG_CANON` | major | Duplicate collapsed, but the *worse* copy was kept as canonical | `pipeline_v6b/util/tagging.py` (canonical-selection rule) |

### Dimension: `chunking` — owner stage `silver_chunk_semantic`
| Type | Severity | Meaning | Owning code file(s) |
|---|---|---|---|
| `CHUNK_ORPHAN_HEADER` | major | A header with no body, or body split away from its header | `ingestion/chunker_v4.py` |
| `CHUNK_OVERSIZED` | major | Chunk far exceeds the token target (coincides with `no_oversized` check) | `ingestion/chunker_v4.py` |
| `CHUNK_FRAGMENTED` | minor | One coherent section shattered into many tiny chunks | `ingestion/chunker_v4.py` |
| `CHUNK_MIDSENTENCE` | minor | Boundary falls mid-sentence / mid-table-row | `ingestion/chunker_v4.py` |
| `CHUNK_MERGED_TOPICS` | minor | Two unrelated sections merged into one chunk | `ingestion/chunker_v4.py` |

### Dimension: `fidelity` — owner stage `bronze` / `silver_modal` (judge vs **source PDF**)
| Type | Severity | Meaning | Owning code file(s) |
|---|---|---|---|
| `FID_TABLE_LOST` | blocker | A table present in the PDF is missing/garbled in the text | `assets/silver_modal.py`, `assets/bronze_blocks.py` |
| `FID_IMAGE_LOST` | minor | Informative figure dropped with no description (logos excepted) | `assets/silver_modal.py` |
| `FID_HEADER_BROKEN` | major | Header hierarchy in the PDF mangled/flattened in the text | `assets/bronze_blocks.py` |
| `FID_CONTENT_DROPPED` | blocker | Substantive prose present in the PDF absent from the text | `assets/bronze_blocks.py` |
| `FID_HALLUCINATION` | blocker | Text contains content **not** in the source PDF | `assets/silver_modal.py` |
| `FID_REPLACEMENT_CHARS` | minor | U+FFFD / decode garbage survived into the text | `assets/bronze_blocks.py` (`clean_replacement_chars`) |

## Per-dimension verdict

Each dimension gets a **binary** verdict — `pass` or `fail` — plus a one-line rationale
and zero or more error tags. (Binary, not 0–3: cheaper and far less run-to-run noise than
an ordinal — see the harness design notes.) A dimension `fail` requires at least one error
tag of that dimension. `blocker` tags force the dimension to `fail`.

## JSON shape the scorer emits (per sample)

```json
{
  "stem": "202611_CSCE_608_600_46648",
  "taxonomy_version": "1.0.0",
  "dimensions": {
    "boilerplate": {"verdict": "pass", "rationale": "...", "findings": []},
    "dedup": {"verdict": "fail", "rationale": "...", "findings": [
      {"type": "DUP_WRONG_CANON", "severity": "major",
       "stage_hypotheses": [{"stage": "silver_tag", "file": "util/tagging.py", "confidence": 0.7}],
       "evidence": "chunk 14 kept the truncated copy; chunk 9 was fuller"}
    ]},
    "chunking": {"verdict": "pass", "rationale": "...", "findings": []},
    "fidelity": {"verdict": "fail", "rationale": "...", "findings": [
      {"type": "FID_TABLE_LOST", "severity": "blocker",
       "stage_hypotheses": [{"stage": "silver_modal", "file": "assets/silver_modal.py", "confidence": 0.8}],
       "evidence": "PDF p.3 grading table absent from processed text"}
    ]}
  }
}
```
