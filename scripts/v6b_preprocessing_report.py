#!/usr/bin/env python
"""Aggregate Inspect judge logs into a scoreboard + prioritized fix list + run comparison.

Reads the latest `.eval` log under each iteration dir's `logs/` and emits, into the first
(latest) iteration dir:

    scoreboard.jsonl   per-sample per-dimension pass/fail + findings (machine-readable)
    summary.md         per-dimension pass rates, error classes ranked by frequency x
                       severity (mapped to the owning code file), judge-noise band (if the
                       log used --epochs), and a paired cross-run comparison (if 2 dirs)

Usage (in container, from /workspace):

    # single run — scoreboard + prioritized fix list
    python scripts/v6b_preprocessing_report.py data/syllabi/_preprocessing_lab/iter_02_<sha>

    # cross-run comparison (current first, previous second)
    python scripts/v6b_preprocessing_report.py <iter_02_dir> <iter_01_dir>

This is deterministic (no LLM). The judge's per-dimension verdict lives in each sample's
metadata["_judge_verdict"]; scores are CORRECT(pass)/INCORRECT(fail) per dimension.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

from tamubot.evals.preprocessing_judge.taxonomy import (
    DIMENSIONS,
    ERROR_TYPES,
    SEVERITY_WEIGHT,
    owners,
    severity,
)

PASS = "C"  # Inspect CORRECT value


def latest_log(iter_dir: Path) -> Path:
    logs = sorted(glob.glob(str(iter_dir / "logs" / "*.eval")))
    if not logs:
        raise SystemExit(f"no .eval log under {iter_dir}/logs — run `inspect eval` first")
    return Path(logs[-1])


def dim_pass(sample, dim: str) -> bool | None:
    sc = sample.scores.get(f"dim_{dim}")
    if sc is None:
        return None
    return str(sc.value) == PASS


def load_run(iter_dir: Path) -> dict:
    """Return {stem: {dim: pass_bool}}, findings list, and epoch-grouped values."""
    ev = read_eval_log(str(latest_log(iter_dir)))
    by_stem: dict[str, dict[str, bool]] = {}
    epochs: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    findings: list[dict] = []
    for s in ev.samples:
        stem = s.id
        verdict = (s.metadata or {}).get("_judge_verdict") or {}
        for dim in DIMENSIONS:
            p = dim_pass(s, dim)
            if p is None:
                continue
            epochs[stem][dim].append(p)
            by_stem.setdefault(stem, {})[dim] = p  # last epoch wins for the point estimate
            d = (verdict.get("dimensions") or {}).get(dim) or {}
            for f in d.get("findings") or []:
                t = f.get("type", "?")
                findings.append({"stem": stem, "dim": dim, "type": t,
                                 "severity": f.get("severity", severity(t)),
                                 "valid": t in ERROR_TYPES, "evidence": f.get("evidence", "")})
    return {"run_id": iter_dir.name, "by_stem": by_stem, "epochs": epochs, "findings": findings,
            "n": len(by_stem)}


def pass_rates(by_stem: dict) -> dict[str, float]:
    out = {}
    for dim in DIMENSIONS:
        vals = [v[dim] for v in by_stem.values() if dim in v]
        out[dim] = round(sum(vals) / len(vals), 3) if vals else None
    return out


def noise_band(epochs: dict) -> dict[str, float]:
    """Per-dimension flip rate across epochs: fraction of stems whose epochs disagree."""
    out = {}
    for dim in DIMENSIONS:
        flips = total = 0
        for stem, dims in epochs.items():
            vals = dims.get(dim, [])
            if len(vals) > 1:
                total += 1
                flips += 1 if len(set(vals)) > 1 else 0
        out[dim] = round(flips / total, 3) if total else None
    return out


def rank_errors(findings: list[dict]) -> list[dict]:
    agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "stems": set()})
    for f in findings:
        if not f["valid"]:
            continue
        a = agg[f["type"]]
        a["count"] += 1
        a["stems"].add(f["stem"])
    rows = []
    for t, a in agg.items():
        w = SEVERITY_WEIGHT.get(severity(t), 1)
        rows.append({"type": t, "severity": severity(t), "count": a["count"],
                     "n_stems": len(a["stems"]), "score": a["count"] * w,
                     "owners": owners(t)})
    return sorted(rows, key=lambda r: (-r["score"], -r["count"]))


def paired_compare(cur: dict, prev: dict, band: dict) -> dict:
    """McNemar-style paired comparison of per-dimension pass on shared stems."""
    try:
        from scipy.stats import binomtest
    except Exception:
        binomtest = None
    shared = sorted(set(cur["by_stem"]) & set(prev["by_stem"]))
    out = {"n_shared": len(shared), "dimensions": {}}
    for dim in DIMENSIONS:
        improved = regressed = 0  # fail->pass / pass->fail
        for stem in shared:
            c, p = cur["by_stem"][stem].get(dim), prev["by_stem"][stem].get(dim)
            if c is None or p is None:
                continue
            if c and not p:
                improved += 1
            elif p and not c:
                regressed += 1
        disc = improved + regressed
        pval = (binomtest(min(improved, regressed), disc).pvalue
                if binomtest and disc else None)
        net = improved - regressed
        bandv = band.get(dim)
        note = "within judge-noise band" if (bandv is not None and shared and
                abs(net) / len(shared) <= bandv) else "exceeds noise band"
        out["dimensions"][dim] = {"improved": improved, "regressed": regressed,
                                  "net": net, "p_value": pval, "noise_note": note}
    return out


def write_summary(cur: dict, band: dict, ranked: list[dict], cross: dict | None,
                  out_dir: Path) -> Path:
    L = [f"# Preprocessing judge summary — {cur['run_id']}",
         f"\nSyllabi judged: **{cur['n']}**\n", "## Per-dimension pass rate\n",
         "| dimension | pass rate |", "|---|---|"]
    pr = pass_rates(cur["by_stem"])
    for dim in DIMENSIONS:
        L.append(f"| {dim} | {pr[dim]} |")
    if any(v is not None for v in band.values()):
        L += ["\n## Judge-noise band (epochs flip rate)\n",
              "Cross-run deltas smaller than this are not distinguishable from judge noise.\n",
              "| dimension | flip rate |", "|---|---|"]
        L += [f"| {dim} | {band[dim]} |" for dim in DIMENSIONS]
    L += ["\n## Systemic error classes (ranked freq x severity) -> owning file\n",
          "| rank | type | sev | count | stems | owner file(s) |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        L.append(f"| {i} | {r['type']} | {r['severity']} | {r['count']} | "
                 f"{r['n_stems']} | {', '.join(r['owners'])} |")
    if not ranked:
        L.append("| – | (no findings) | – | – | – | – |")
    if cross:
        L += [f"\n## Cross-run comparison vs {cross['_prev']} ({cross['n_shared']} shared)\n",
              "| dimension | improved | regressed | net | p | note |",
              "|---|---|---|---|---|---|"]
        for dim in DIMENSIONS:
            d = cross["dimensions"][dim]
            L.append(f"| {dim} | {d['improved']} | {d['regressed']} | {d['net']} | "
                     f"{d['p_value']} | {d['noise_note']} |")
    L += ["\n---", "_Findings are judge hypotheses. Confirm the top class against the source "
          "PDF (and golden recall@5) before editing the owning file. Log the fix in "
          "docs/pipeline-failures.md._"]
    path = out_dir / "summary.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("iter_dirs", nargs="+", help="iteration dir(s); current first, previous second")
    args = ap.parse_args()
    dirs = [Path(d) for d in args.iter_dirs]
    cur = load_run(dirs[0])
    band = noise_band(cur["epochs"])
    ranked = rank_errors(cur["findings"])

    # scoreboard.jsonl
    sb = dirs[0] / "scoreboard.jsonl"
    with sb.open("w", encoding="utf-8") as fh:
        for stem, dims in sorted(cur["by_stem"].items()):
            fh.write(json.dumps({"stem": stem, "dimensions": dims,
                                 "findings": [f for f in cur["findings"] if f["stem"] == stem]}) + "\n")

    cross = None
    if len(dirs) > 1:
        prev = load_run(dirs[1])
        cross = paired_compare(cur, prev, band)
        cross["_prev"] = prev["run_id"]

    summary = write_summary(cur, band, ranked, cross, dirs[0])
    print(f"scoreboard: {sb}")
    print(f"summary:    {summary}")
    print(f"\nPass rates: {pass_rates(cur['by_stem'])}")
    if ranked:
        top = ranked[0]
        print(f"Top error class: {top['type']} ({top['severity']}, n={top['count']}) "
              f"-> {', '.join(top['owners'])}")


if __name__ == "__main__":
    main()
