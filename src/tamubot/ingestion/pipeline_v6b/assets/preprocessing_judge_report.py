"""v6b_preprocessing_judge_report: surface the preprocessing-judge results in the Dagster
UI. REPORT ONLY — reads the latest iteration's scoreboard.jsonl (produced by the judge,
sub-agent or model-graded) and emits per-dimension pass rates, the top systemic error
class mapped to its owning code file, and a clickable link to the Inspect log viewer.

This is the bridge between the Inspect-based improvement loop and Dagster: it does NOT run
the judge (sub-agents are driven from Claude Code; the optional model-graded path runs via
`inspect eval`). It consumes whatever scoreboard the loop last wrote.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from dagster import MaterializeResult, MetadataValue, asset

from tamubot.evals.preprocessing_judge.taxonomy import (
    DIMENSIONS,
    SEVERITY_WEIGHT,
    owners,
    severity,
)
from tamubot.ingestion.pipeline_v5.util import DATA_ROOT

LAB_ROOT = Path(DATA_ROOT) / "_preprocessing_lab"
# Where the user serves `inspect view` (see iterate-preprocessing skill); 3001 is the
# free host-published port. Override with PREPROC_VIEW_URL.
VIEW_URL = os.getenv("PREPROC_VIEW_URL", "http://localhost:3001")
# Container /workspace == host repo root; used to turn the container-side artifact paths
# into Windows links/paths the Dagster browser + File Explorer can actually open.
HOST_ROOT = os.getenv("PREPROC_HOST_ROOT", "C:/dev/TAMUBOT")


def _host(relpath: str) -> tuple[str, str]:
    """repo-relative path -> (file:// url, Windows backslash path)."""
    rp = relpath.replace("\\", "/").replace("/workspace/", "").lstrip("/")
    return f"file:///{HOST_ROOT}/{rp}", f"{HOST_ROOT}/{rp}".replace("/", "\\")


def _latest_scoreboard() -> Path | None:
    boards = sorted(LAB_ROOT.glob("*/scoreboard.jsonl"), key=lambda p: p.stat().st_mtime)
    return boards[-1] if boards else None


def _summarize(scoreboard: Path) -> dict:
    rows = [json.loads(ln) for ln in scoreboard.read_text(encoding="utf-8").splitlines() if ln.strip()]
    pass_counts = {d: [0, 0] for d in DIMENSIONS}  # [passed, total]
    err_agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "stems": set()})
    for r in rows:
        for d, passed in (r.get("dimensions") or {}).items():
            if d in pass_counts:
                pass_counts[d][0] += int(bool(passed))
                pass_counts[d][1] += 1
        for f in r.get("findings") or []:
            t = f.get("type")
            if t and t != "?":
                err_agg[t]["count"] += 1
                err_agg[t]["stems"].add(r.get("stem"))
    rates = {d: (round(p / t, 3) if t else None) for d, (p, t) in pass_counts.items()}
    ranked = sorted(
        ({"type": t, "severity": severity(t), "count": a["count"],
          "n_stems": len(a["stems"]), "score": a["count"] * SEVERITY_WEIGHT.get(severity(t), 1),
          "owners": owners(t)} for t, a in err_agg.items()),
        key=lambda r: (-r["score"], -r["count"]),
    )
    return {"run_id": scoreboard.parent.name, "n": len(rows), "pass_rates": rates,
            "ranked": ranked, "dir": scoreboard.parent}


def _compute(context) -> MaterializeResult:  # context left unannotated (see note below)
    sb = _latest_scoreboard()
    if sb is None:
        context.log.warning("no scoreboard.jsonl under %s — run the judge first", LAB_ROOT)
        return MaterializeResult(metadata={
            "status": "no judge runs yet",
            "hint": "build pairs + run the judge (see iterate-preprocessing skill)",
        })

    s = _summarize(sb)
    top = s["ranked"][0] if s["ranked"] else None
    index_html = s["dir"] / "index.html"
    summary_md = s["dir"] / "summary.md"

    md = summary_md.read_text(encoding="utf-8") if summary_md.exists() else "_(no summary.md)_"
    context.log.info("judge report %s: %d syllabi, pass rates %s",
                     s["run_id"], s["n"], s["pass_rates"])

    review_url, review_win = _host(str(index_html))
    _, sb_win = _host(str(sb))
    meta: dict = {
        "run_id": s["run_id"],
        "syllabi_judged": s["n"],
        # the full report, rendered inline right here in the Dagster UI:
        "report": MetadataValue.md(md),
        # clickable: open the per-syllabus judge reasoning in the Inspect viewer
        # (start it first: inspect view --log-dir <iter>/logs --host 0.0.0.0 --port 3001)
        "open_in_inspect_viewer": MetadataValue.url(VIEW_URL),
        # clickable file:// link to the side-by-side review page (opens in a browser tab)
        "open_side_by_side_review": MetadataValue.url(review_url),
        # copy-paste into File Explorer if the file:// link is blocked by the browser
        "side_by_side_path": review_win,
        "scoreboard_path": sb_win,
    }
    for d in DIMENSIONS:
        meta[f"pass_{d}"] = MetadataValue.float(s["pass_rates"][d]) if s["pass_rates"][d] is not None else "n/a"
    if top:
        meta["top_error_class"] = f"{top['type']} ({top['severity']}, n={top['count']})"
        meta["top_error_owner"] = ", ".join(top["owners"])
    return MaterializeResult(metadata=meta)


v6b_preprocessing_judge_report = asset(
    name="v6b_preprocessing_judge_report",
    group_name="v6b_ops",
    description=(
        "Report-only: surfaces the latest preprocessing-judge iteration (per-dimension "
        "pass rates, top error class -> owning file, link to the Inspect log viewer) in "
        "the Dagster UI. Consumes the scoreboard written by the judge loop."
    ),
)(_compute)
