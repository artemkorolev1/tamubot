"""Canonical brief handed to each Claude Code sub-agent judging one before/after pair.

The orchestrator (iterate-preprocessing skill) spawns one general-purpose sub-agent per
sample with `build_brief(...)`. The agent reads the files (including the PDF page PNGs,
which Read renders visually), applies the frozen taxonomy, and writes ONLY the verdict
JSON to <verdicts_dir>/<stem>.json. `preprocessing_judge_subagent` then loads those into
the Inspect viewer + scorers + report.

`python -m tamubot.evals.preprocessing_judge.agent_brief <samples.jsonl> <verdicts_dir>`
prints one brief per stem (separated by a marker) for inspection / scripted dispatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tamubot.evals.preprocessing_judge.prompts import SYSTEM_PROMPT


def build_brief(meta: dict, verdicts_dir: str) -> str:
    stem = meta.get("stem")
    pages = meta.get("pdf_pages", []) or []
    page_lines = "\n".join(f"  - {p}" for p in pages) or "  (none)"
    anchors = json.dumps(meta.get("check_anchors", {}), indent=2)
    out_path = str(Path(verdicts_dir) / f"{stem}.json")
    return f"""You are judging the preprocessing of ONE TAMU syllabus: {stem}.

{SYSTEM_PROMPT}

## Your inputs (read all of them)
- ORIGINAL (faithful extracted text):   {meta.get('original_md')}
- PROCESSED (what RAG sees; hidden chunks removed):   {meta.get('processed_md')}
- WHY chunks were hidden (boilerplate/duplicate provenance):   {meta.get('why_json')}
- SOURCE PDF page images — Read EACH of these (they render visually); judge `fidelity`
  strictly against them:
{page_lines}

## Deterministic check anchors (authoritative for rates/counts — do NOT re-estimate)
{anchors}

## Your output (the ONLY thing you write)
Write the verdict JSON object — exactly the shape specified above, nothing else — to:
    {out_path}
Use the Write tool to create that file. Do not print the JSON as your final message;
write the file, then reply with a one-line confirmation. Do not modify any other file.
"""


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: agent_brief <samples.jsonl> <verdicts_dir>")
    samples, verdicts_dir = sys.argv[1], sys.argv[2]
    for line in Path(samples).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)["metadata"]
        print(f"\n===== BRIEF: {meta.get('stem')} =====")
        print(build_brief(meta, verdicts_dir))


if __name__ == "__main__":
    main()
