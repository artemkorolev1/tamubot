"""Inspect AI judge for v6b preprocessing quality.

One judge model call per syllabus produces a per-dimension binary verdict against the
frozen taxonomy (docs/preprocessing_error_taxonomy.md). The model generation IS the judge
(visible inline in `inspect view`); the four scorers just parse and tally it, so there is
no extra model call per dimension.

Run (inside the container, from /workspace):

    inspect eval src/tamubot/evals/preprocessing_judge/task.py@preprocessing_judge \\
        -T samples=data/syllabi/_preprocessing_lab/iter_01_<sha>/samples.jsonl \\
        --model google/gemini-2.5-flash --log-dir <iter_dir>/logs

    # judge-noise calibration (Phase 0): repeat each sample 3x
    inspect eval ...@preprocessing_judge -T samples=... --epochs 3 --model ...

    # blind cross-run A/B (current vs previous iteration)
    inspect eval ...@preprocessing_pairwise \\
        -T current=<iterN>/samples.jsonl -T previous=<iterN-1>/samples.jsonl --model ...
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

# repo-root anchor for the relative paths stored in samples.jsonl (set to /workspace for
# in-container Inspect runs; defaults to cwd, which is also /workspace there)
_ROOT = Path(os.environ.get("PREPROC_ROOT", "."))

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
    ModelOutput,
)
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, TaskState, solver

from tamubot.evals.preprocessing_judge.prompts import (
    PAIRWISE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

DIMENSIONS = ["boilerplate", "dedup", "chunking", "fidelity"]
_VERDICT_KEY = "_judge_verdict"


# --------------------------------------------------------------------- helpers
def _resolve(path: str) -> Path:
    """Resolve a stored (relative) path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def _read(path: str | None, limit: int = 60_000) -> str:
    if not path:
        return ""
    p = _resolve(path)
    if not p.exists():
        return f"<missing: {path}>"
    return p.read_text(encoding="utf-8", errors="replace")[:limit]


def _images(page_paths: list[str], max_pages: int) -> list[ContentImage]:
    out: list[ContentImage] = []
    for pp in (page_paths or [])[:max_pages]:
        rp = _resolve(pp)
        if rp.exists():
            out.append(ContentImage(image=str(rp)))
    return out


def _parse_verdict(completion: str) -> dict:
    """Best-effort extraction of the judge JSON object."""
    text = (completion or "").strip()
    if "```" in text:  # strip code fences
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.removeprefix("json").strip()
    if "{" in text and "}" in text:
        text = text[text.index("{") : text.rindex("}") + 1]
    try:
        return json.loads(text)
    except Exception:
        try:
            import json_repair  # available in container

            return json_repair.loads(text)
        except Exception:
            return {}


def _dim_pass(verdict: dict, dim: str) -> tuple[bool, str, list]:
    """Returns (passed, rationale, findings) with blocker-forces-fail enforced."""
    d = (verdict.get("dimensions") or {}).get(dim) or {}
    findings = d.get("findings") or []
    has_blocker = any((f or {}).get("severity") == "blocker" for f in findings)
    passed = d.get("verdict") == "pass" and not has_blocker
    return passed, d.get("rationale", ""), findings


# ------------------------------------------------------------------- messages
def build_judge_messages(m: dict, max_pages: int) -> list:
    """Multimodal judge prompt (system + user with text + PDF page images).

    Shared by the model-graded solver and the sub-agent verdict loader so the Inspect
    viewer shows the SAME interpretable context (original/processed/PDF) either way.
    """
    anchors = json.dumps(m.get("check_anchors", {}), indent=2)
    why = _read(m.get("why_json"))
    user_text = (
        f"STEM: {m.get('stem')}  (dept {m.get('dept')})\n\n"
        f"DETERMINISTIC CHECK ANCHORS (authoritative for rates/counts):\n{anchors}\n\n"
        f"WHY chunks were hidden from PROCESSED (boilerplate/duplicate provenance):\n"
        f"{why}\n\n"
        f"===== ORIGINAL (faithful extracted text) =====\n{_read(m.get('original_md'))}\n\n"
        f"===== PROCESSED (RAG-visible; hidden chunks removed) =====\n"
        f"{_read(m.get('processed_md'))}\n\n"
        f"Below: SOURCE PDF page images (fidelity reference)."
    )
    content = [ContentText(text=user_text)] + _images(m.get("pdf_pages", []), max_pages)
    return [ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content=content)]


# ---------------------------------------------------------------------- solver
@solver
def judge_solver(max_pages: int = 12):
    """Model-graded path: assemble the prompt and grade once with the --model LLM."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages = build_judge_messages(state.metadata, max_pages)
        state = await generate(state)
        state.metadata[_VERDICT_KEY] = _parse_verdict(state.output.completion)
        return state

    return solve


@solver
def verdict_loader(verdicts_dir: str, max_pages: int = 12):
    """Sub-agent path: load a verdict JSON produced by a Claude Code sub-agent (no model
    call) so it renders in the same Inspect viewer and feeds the same scorers/report.

    Each sub-agent writes <verdicts_dir>/<stem>.json (the taxonomy verdict object). Run
    the task with `--model mockllm/model` — no API key, no provider needed.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        m = state.metadata
        raw = _read(str(Path(verdicts_dir) / f"{m.get('stem')}.json"))
        if raw.startswith("<missing"):
            raw = "{}"  # un-judged stem → scorers mark unparseable
        # same interpretable context in the viewer; output = the sub-agent's verdict
        state.messages = build_judge_messages(m, max_pages)
        state.output = ModelOutput.from_content("subagent", raw)
        state.metadata[_VERDICT_KEY] = _parse_verdict(raw)
        return state

    return solve


# --------------------------------------------------------------------- scorers
def _dimension_scorer(dim: str):
    @scorer(name=f"dim_{dim}", metrics=[accuracy(), stderr()])
    def make():
        async def compute(state: TaskState, target: Target) -> Score:
            verdict = state.metadata.get(_VERDICT_KEY) or {}
            if not verdict:
                return Score(value=INCORRECT, explanation="judge returned unparseable JSON")
            passed, rationale, findings = _dim_pass(verdict, dim)
            tags = ",".join(f.get("type", "?") for f in findings) or "—"
            return Score(
                value=CORRECT if passed else INCORRECT,
                answer="pass" if passed else "fail",
                explanation=f"{rationale}  [{tags}]",
                metadata={"findings": findings},
            )

        return compute

    return make()


# ----------------------------------------------------------------------- tasks
def _samples_from_jsonl(path: str) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


@task
def preprocessing_judge(samples: str, max_pages: int = 12, split: str | None = None) -> Task:
    """Absolute per-syllabus judge over a samples.jsonl built by
    scripts/v6b_build_comparison_pairs.py."""
    rows = _samples_from_jsonl(samples)
    if split:
        rows = [r for r in rows if (r.get("metadata") or {}).get("split") == split]
    ds = MemoryDataset([
        Sample(input=r["input"], target=r.get("target", ""), id=r["id"], metadata=r["metadata"])
        for r in rows
    ])
    return Task(
        dataset=ds,
        solver=judge_solver(max_pages=max_pages),
        scorer=[_dimension_scorer(d) for d in DIMENSIONS],
    )


@task
def preprocessing_judge_subagent(samples: str, verdicts_dir: str, max_pages: int = 12,
                                 split: str | None = None) -> Task:
    """Sub-agent judging path (no model backend). Claude Code sub-agents each evaluate one
    before/after pair and write <verdicts_dir>/<stem>.json; this task loads those verdicts
    into the Inspect viewer + scorers + report. Run with `--model mockllm/model`."""
    rows = _samples_from_jsonl(samples)
    if split:
        rows = [r for r in rows if (r.get("metadata") or {}).get("split") == split]
    ds = MemoryDataset([
        Sample(input=r["input"], target=r.get("target", ""), id=r["id"], metadata=r["metadata"])
        for r in rows
    ])
    return Task(
        dataset=ds,
        solver=verdict_loader(verdicts_dir, max_pages=max_pages),
        scorer=[_dimension_scorer(d) for d in DIMENSIONS],
    )


@task
def preprocessing_pairwise(current: str, previous: str, max_pages: int = 12,
                           seed: int = 42) -> Task:
    """Blind, randomized A/B of two runs' PROCESSED views over their shared stems.
    Primary cross-run instrument: far more stable than diffing absolute scores."""
    cur = {r["id"]: r for r in _samples_from_jsonl(current)}
    prev = {r["id"]: r for r in _samples_from_jsonl(previous)}
    rng = random.Random(seed)
    samples: list[Sample] = []
    for stem in sorted(set(cur) & set(prev)):
        cm, pm = cur[stem]["metadata"], prev[stem]["metadata"]
        cur_is_a = rng.random() < 0.5  # blind assignment
        a, b = (cm, pm) if cur_is_a else (pm, cm)
        samples.append(Sample(
            input=f"Pairwise A/B for {stem}", id=stem,
            metadata={**cm, "a_processed": a["processed_md"], "b_processed": b["processed_md"],
                      "current_is_a": cur_is_a},
        ))
    return Task(dataset=MemoryDataset(samples), solver=pairwise_solver(max_pages=max_pages),
                scorer=[_pairwise_scorer(d) for d in DIMENSIONS])


@solver
def pairwise_solver(max_pages: int = 12):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        m = state.metadata
        text = (
            f"STEM: {m.get('stem')}\n\n===== ORIGINAL =====\n{_read(m.get('original_md'))}\n\n"
            f"===== VIEW A =====\n{_read(m.get('a_processed'))}\n\n"
            f"===== VIEW B =====\n{_read(m.get('b_processed'))}\n\n"
            f"PDF pages follow. Decide better view per dimension; JSON only."
        )
        state.messages = [
            ChatMessageSystem(content=PAIRWISE_SYSTEM_PROMPT),
            ChatMessageUser(content=[ContentText(text=text)] + _images(m.get("pdf_pages", []), max_pages)),
        ]
        state = await generate(state)
        state.metadata[_VERDICT_KEY] = _parse_verdict(state.output.completion)
        return state

    return solve


def _pairwise_scorer(dim: str):
    @scorer(name=f"ab_{dim}", metrics=[accuracy(), stderr()])
    def make():
        async def compute(state: TaskState, target: Target) -> Score:
            v = state.metadata.get(_VERDICT_KEY) or {}
            choice = ((v.get("dimensions") or {}).get(dim) or {}).get("better", "tie")
            cur_is_a = state.metadata.get("current_is_a")
            # CORRECT == current run won this dimension; tie/loss == INCORRECT
            current_won = (choice == "A" and cur_is_a) or (choice == "B" and not cur_is_a)
            return Score(value=CORRECT if current_won else INCORRECT, answer=choice,
                         explanation=((v.get("dimensions") or {}).get(dim) or {}).get("reason", ""))

        return compute

    return make()
