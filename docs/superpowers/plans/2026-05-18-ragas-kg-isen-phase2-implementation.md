# Ragas KG ISEN Phase 2 — Persona-Driven QA Generation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plumb a `Course-Shopping Student` persona and a durable-attribute prompt nudge into the existing Ragas testset generator, then run Phase 2 to produce a 20-item `balanced_50_50` golden set against the already-cached KG.

**Architecture:** Add a YAML-defined persona, a small loader module, and a `--persona-file` CLI flag to the existing generator script. Mutate the `generate_query_reference_prompt.instruction` on `SingleHopSpecificQuerySynthesizer` at build time using Ragas's `get_prompts` / `set_prompts` API. No KG rebuild, no schema changes, no new synthesizers.

**Tech Stack:** Python 3.14, Ragas 0.4.3 (`ragas.testset.persona.Persona`, `SingleHopSpecificQuerySynthesizer`, `MultiHopAbstractQuerySynthesizer`, `TestsetGenerator`), PyYAML, pytest, Streamlit (downstream consumer of the XLSX). All LLM calls flow through the TAMU OpenAI-compatible gateway via `tamu_openai_workaround.wrap_for_tamu`.

**Spec:** `docs/superpowers/specs/2026-05-18-ragas-kg-isen-phase2-design.md`

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Modify | Pin `pyyaml` (currently transitive) |
| `tamu_data/evals/personas/course_shopping_student.yaml` | Create | Single persona definition, editable without code change |
| `src/tamubot/evals/personas.py` | Create | YAML → `list[ragas.testset.persona.Persona]` loader. Fails loud on missing/empty |
| `src/tamubot/evals/generate_ragas_testset.py` | Modify | Add `--persona-file` flag; apply prompt nudge in `build_query_distribution`; thread `persona_list` through `generate_testset` and `main` |
| `tests/test_personas.py` | Create | Unit tests for loader; unit test that the single-hop synthesizer carries the durable-attribute nudge after `build_query_distribution` runs |

Cached KG (`tamu_data/evals/ragas_kg.json`) and the v4 corpus (`data/syllabi/silver/06_chunk/`) are reused as-is.

---

### Task 1: Pin PyYAML

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Confirm `pyyaml` is currently usable but unpinned**

Run: `python -c "import yaml; print(yaml.__version__)" && grep -i ^yaml /workspace/requirements.txt`

Expected: prints a version like `6.0.3`, and the `grep` returns nothing (PyYAML is transitive only).

- [ ] **Step 2: Add the pin**

Append to `requirements.txt` in the same section as `rapidfuzz` (the Ragas-driven adds). Use the exact line:

```
pyyaml>=6.0,<7.0
```

- [ ] **Step 3: Reinstall to make sure the pin resolves**

Run: `pip install -r requirements.txt 2>&1 | tail -5`

Expected: pip prints `Requirement already satisfied: pyyaml...` with no error.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps(evals): pin pyyaml for persona YAML loader"
```

---

### Task 2: Create the persona YAML

**Files:**
- Create: `tamu_data/evals/personas/course_shopping_student.yaml`

- [ ] **Step 1: Create the personas directory**

Run: `mkdir -p /workspace/tamu_data/evals/personas`

Expected: no output, directory exists.

- [ ] **Step 2: Write the persona file**

Create `tamu_data/evals/personas/course_shopping_student.yaml` with this exact content:

```yaml
# Persona for Phase 2 Ragas QA generation.
# See docs/superpowers/specs/2026-05-18-ragas-kg-isen-phase2-design.md
personas:
  - name: Course-Shopping Student
    role_description: |
      A Texas A&M student browsing ISEN syllabi to decide which courses to take
      next semester. Asks two kinds of questions:
      (1) Factual lookups about a single course — content/topics covered,
          learning outcomes, schedule and weekly topic flow, tools/software/
          textbooks used, grading breakdown, exam structure, modality,
          attendance policy, listed prerequisites.
      (2) Comparisons across courses — which courses cover a given topic,
          which use a particular tool, how two courses differ in workload or
          focus.
      Does NOT ask personalized "given my background, am I ready for X"
      questions.
```

- [ ] **Step 3: Verify YAML parses**

Run:
```bash
python -c "
import yaml, pathlib
data = yaml.safe_load(pathlib.Path('/workspace/tamu_data/evals/personas/course_shopping_student.yaml').read_text())
assert 'personas' in data, 'missing personas key'
assert len(data['personas']) == 1, 'expected one persona'
p = data['personas'][0]
assert p['name'] == 'Course-Shopping Student', p['name']
assert 'browsing ISEN syllabi' in p['role_description']
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add tamu_data/evals/personas/course_shopping_student.yaml
git commit -m "data(evals): add Course-Shopping Student persona for Phase 2"
```

---

### Task 3: Persona loader module — write the failing test

**Files:**
- Create: `tests/test_personas.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_personas.py` with this exact content:

```python
"""Unit tests for tamubot.evals.personas.load_personas."""
from pathlib import Path

import pytest


def test_load_single_persona_from_yaml(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    yaml_text = (
        "personas:\n"
        "  - name: Test Persona\n"
        "    role_description: |\n"
        "      A student deciding between courses.\n"
    )
    p = tmp_path / "p.yaml"
    p.write_text(yaml_text)

    personas = load_personas(p)

    assert len(personas) == 1
    assert personas[0].name == "Test Persona"
    assert "A student deciding between courses." in personas[0].role_description


def test_load_personas_returns_ragas_persona_instances(tmp_path: Path) -> None:
    from ragas.testset.persona import Persona

    from tamubot.evals.personas import load_personas

    p = tmp_path / "p.yaml"
    p.write_text("personas:\n  - name: X\n    role_description: y\n")

    personas = load_personas(p)

    assert isinstance(personas[0], Persona)


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    with pytest.raises(FileNotFoundError):
        load_personas(tmp_path / "does_not_exist.yaml")


def test_empty_personas_list_raises_valueerror(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    p = tmp_path / "p.yaml"
    p.write_text("personas: []\n")

    with pytest.raises(ValueError, match="at least one persona"):
        load_personas(p)


def test_missing_personas_key_raises_valueerror(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    p = tmp_path / "p.yaml"
    p.write_text("other_key: value\n")

    with pytest.raises(ValueError, match="personas"):
        load_personas(p)


def test_load_multiple_personas(tmp_path: Path) -> None:
    from tamubot.evals.personas import load_personas

    yaml_text = (
        "personas:\n"
        "  - name: A\n"
        "    role_description: alpha\n"
        "  - name: B\n"
        "    role_description: beta\n"
    )
    p = tmp_path / "p.yaml"
    p.write_text(yaml_text)

    personas = load_personas(p)

    assert [pp.name for pp in personas] == ["A", "B"]
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `pytest tests/test_personas.py -v`

Expected: all six tests `ERROR` or `FAIL` with `ModuleNotFoundError: No module named 'tamubot.evals.personas'`.

---

### Task 4: Persona loader module — implement

**Files:**
- Create: `src/tamubot/evals/personas.py`

- [ ] **Step 1: Write the implementation**

Create `src/tamubot/evals/personas.py` with this exact content:

```python
"""Load Ragas Personas from a YAML file.

The YAML schema:

    personas:
      - name: <str>
        role_description: <str>
      - name: <str>
        role_description: <str>
"""
from __future__ import annotations

from pathlib import Path

import yaml
from ragas.testset.persona import Persona


def load_personas(path: Path) -> list[Persona]:
    """Load Personas from a YAML file. Fails loud on missing/malformed input."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Persona file not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if "personas" not in data:
        raise ValueError(f"{path} is missing the top-level 'personas' key")

    raw = data["personas"]
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"{path} must contain at least one persona under 'personas'")

    personas: list[Persona] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: personas[{i}] is not a mapping")
        name = item.get("name")
        role = item.get("role_description")
        if not name or not role:
            raise ValueError(f"{path}: personas[{i}] missing name or role_description")
        personas.append(Persona(name=str(name), role_description=str(role).strip()))

    return personas
```

- [ ] **Step 2: Run tests, confirm they pass**

Run: `pytest tests/test_personas.py -v`

Expected: all six tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/tamubot/evals/personas.py tests/test_personas.py
git commit -m "feat(evals): YAML persona loader for Ragas testset generation"
```

---

### Task 5: Durable-attribute prompt nudge — write the failing test

**Files:**
- Modify: `tests/test_personas.py`

- [ ] **Step 1: Append the nudge test**

Append this to the end of `tests/test_personas.py`:

```python
# ---------------------------------------------------------------------------
# Prompt-nudge regression: build_query_distribution must attach a durable-
# attribute instruction to the SingleHopSpecificQuerySynthesizer.
# ---------------------------------------------------------------------------


def test_single_hop_synthesizer_carries_durable_attribute_nudge() -> None:
    """The single-hop synthesizer's instruction must mention durable attributes
    and explicitly steer away from term-bound deadlines."""
    from unittest.mock import MagicMock

    from tamubot.evals.generate_ragas_testset import build_query_distribution

    dist = build_query_distribution(llm=MagicMock(), preset="balanced_50_50")

    single_hop, weight = dist[0]
    assert weight == 0.50
    instruction = single_hop.get_prompts()["generate_query_reference_prompt"].instruction

    assert "durable course attributes" in instruction
    assert "term-bound deadlines" in instruction


def test_multi_hop_abstract_synthesizer_is_unmodified() -> None:
    """The multi-hop abstract synthesizer must NOT receive the nudge."""
    from unittest.mock import MagicMock

    from tamubot.evals.generate_ragas_testset import build_query_distribution

    dist = build_query_distribution(llm=MagicMock(), preset="balanced_50_50")

    multi_hop, weight = dist[1]
    assert weight == 0.50
    # Inspect every prompt on the multi-hop synthesizer; none should carry
    # the single-hop-specific nudge marker.
    for prompt in multi_hop.get_prompts().values():
        assert "durable course attributes" not in prompt.instruction
```

- [ ] **Step 2: Run tests, confirm the two new ones fail**

Run: `pytest tests/test_personas.py::test_single_hop_synthesizer_carries_durable_attribute_nudge tests/test_personas.py::test_multi_hop_abstract_synthesizer_is_unmodified -v`

Expected: both FAIL with `AssertionError: assert 'durable course attributes' in instruction` (because the nudge has not been added yet).

---

### Task 6: Durable-attribute prompt nudge — implement

**Files:**
- Modify: `src/tamubot/evals/generate_ragas_testset.py` (function `build_query_distribution`)

- [ ] **Step 1: Add the nudge constant and the helper**

Open `src/tamubot/evals/generate_ragas_testset.py`. Just after the `DISTRIBUTION_PRESETS` line (around line 40), add:

```python
SELECTION_TIME_NUDGE = (
    "\n\nThe student is choosing courses for a future semester and is not "
    "currently enrolled. Prefer questions about durable course attributes — "
    "topics covered, learning outcomes, tools used, grading structure, "
    "prerequisites. Avoid questions about specific term-bound deadlines "
    "(e.g., 'when is the midterm scheduled' or 'what is due next week')."
)


def _apply_selection_time_nudge(synth) -> None:
    """Append SELECTION_TIME_NUDGE to the synthesizer's query-generation prompt."""
    prompts = synth.get_prompts()
    prompt = prompts["generate_query_reference_prompt"]
    prompt.instruction = prompt.instruction + SELECTION_TIME_NUDGE
    synth.set_prompts(**{"generate_query_reference_prompt": prompt})
```

- [ ] **Step 2: Apply the nudge inside `build_query_distribution`**

Replace the body of `build_query_distribution` so every `SingleHopSpecificQuerySynthesizer` instance — in every preset — gets the nudge applied. Replace the entire function with this exact code:

```python
def build_query_distribution(llm, preset: str = "default"):
    """Return a Ragas query_distribution list for the chosen preset.

    Every SingleHopSpecificQuerySynthesizer in the returned list has
    SELECTION_TIME_NUDGE appended to its generate_query_reference_prompt
    instruction.

    Presets:
      - "default":         50% single-hop / 30% multi-hop-specific / 20% multi-hop-abstract
      - "balanced_50_50":  50% single-hop-specific / 50% multi-hop-abstract
      - "semantic_only":   100% multi-hop-abstract (all -> expected_function=semantic_general)
    """
    from ragas.testset.synthesizers.multi_hop.abstract import (
        MultiHopAbstractQuerySynthesizer,
    )
    from ragas.testset.synthesizers.multi_hop.specific import (
        MultiHopSpecificQuerySynthesizer,
    )
    from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )

    def _single_hop():
        s = SingleHopSpecificQuerySynthesizer(llm=llm)
        _apply_selection_time_nudge(s)
        return s

    if preset == "balanced_50_50":
        return [
            (_single_hop(), 0.50),
            (MultiHopAbstractQuerySynthesizer(llm=llm), 0.50),
        ]
    if preset == "semantic_only":
        return [(MultiHopAbstractQuerySynthesizer(llm=llm), 1.0)]
    # default
    return [
        (_single_hop(), 0.50),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 0.30),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 0.20),
    ]
```

- [ ] **Step 3: Run the new nudge tests, confirm they pass**

Run: `pytest tests/test_personas.py -v`

Expected: all eight tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/tamubot/evals/generate_ragas_testset.py tests/test_personas.py
git commit -m "feat(evals): durable-attribute nudge on single-hop synthesizer"
```

---

### Task 7: `--persona-file` CLI flag + threading into `generate_testset`

**Files:**
- Modify: `src/tamubot/evals/generate_ragas_testset.py` (`parse_args`, `generate_testset`, `main`)

- [ ] **Step 1: Add the CLI flag**

In `parse_args()`, just after the `--temperature` argument block (around line 627), add:

```python
    p.add_argument(
        "--persona-file",
        type=str,
        default="tamu_data/evals/personas/course_shopping_student.yaml",
        help=(
            "Path to a persona YAML file (see src/tamubot/evals/personas.py). "
            "Pass an empty string to skip personas and fall back to Ragas defaults."
        ),
    )
```

- [ ] **Step 2: Extend `generate_testset` to accept `persona_list`**

Replace the existing `generate_testset` function with this exact code:

```python
def generate_testset(kg, llm, embedding_model, query_distribution, testset_size: int, persona_list=None):
    """Run RAGAS TestsetGenerator and return a DataFrame.

    If ``persona_list`` is a non-empty list of ragas.testset.persona.Persona
    objects, it is forwarded to ``generator.generate``; otherwise Ragas falls
    back to its internal default persona.
    """
    from ragas.testset import TestsetGenerator

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
        knowledge_graph=kg,
    )
    kwargs = {
        "testset_size": testset_size,
        "query_distribution": query_distribution,
        "raise_exceptions": False,
    }
    if persona_list:
        kwargs["persona_list"] = persona_list
    testset = generator.generate(**kwargs)
    return testset.to_pandas()  # type: ignore[union-attr]
```

- [ ] **Step 3: Load personas and thread through `main`**

In `main()`, just after the line `query_distribution = build_query_distribution(llm, preset=args.distribution)`, add:

```python
    # --- Load personas (optional; empty path = Ragas defaults) ---
    persona_list = None
    if args.persona_file:
        from tamubot.evals.personas import load_personas

        persona_list = load_personas(Path(args.persona_file))
        print(f"Loaded {len(persona_list)} persona(s): {[p.name for p in persona_list]}")
    else:
        print("No persona file provided — falling back to Ragas default persona.")
```

Then, inside the batched generation loop in `main()`, change the call:

```python
            raw = generate_testset(kg, llm, embedding_model, query_distribution, this_batch)
```

to:

```python
            raw = generate_testset(
                kg, llm, embedding_model, query_distribution, this_batch,
                persona_list=persona_list,
            )
```

- [ ] **Step 4: Smoke-test the CLI plumbing with `--dry-run`**

Run:

```bash
cd /workspace && python -m tamubot.evals.generate_ragas_testset \
  --corpus-dir /workspace/data/syllabi/silver/06_chunk \
  --include-terms 202611,202621,202641 \
  --persona-file tamu_data/evals/personas/course_shopping_student.yaml \
  --dry-run
```

Expected: prints `Loaded 1 persona(s): ['Course-Shopping Student']`, lists 75 documents, and exits at `--dry-run: stopping before generation.` No traceback. Zero LLM/embedding calls.

- [ ] **Step 5: Smoke-test the empty-string escape hatch**

Run:

```bash
cd /workspace && python -m tamubot.evals.generate_ragas_testset \
  --corpus-dir /workspace/data/syllabi/silver/06_chunk \
  --include-terms 202611 \
  --persona-file "" \
  --dry-run
```

Expected: prints `No persona file provided — falling back to Ragas default persona.` and exits cleanly.

- [ ] **Step 6: Smoke-test KG load with the cached graph (no generation)**

Run:

```bash
cd /workspace && python -m tamubot.evals.generate_ragas_testset \
  --corpus-dir /workspace/data/syllabi/silver/06_chunk \
  --include-terms 202611,202621,202641 \
  --persona-file tamu_data/evals/personas/course_shopping_student.yaml \
  --provider tamu \
  --build-kg-only
```

Expected: `Loading cached KG from tamu_data/evals/ragas_kg.json`, prints `KG: nodes=210 ({...}), relationships=3252`, then exits at `--build-kg-only`. Zero net new LLM calls because the KG cache hits.

- [ ] **Step 7: Commit**

```bash
git add src/tamubot/evals/generate_ragas_testset.py
git commit -m "feat(evals): plumb --persona-file flag through testset generation"
```

---

### Task 8: Run Phase 2 generation

This task burns ~50–80 TAMU chat completions and a few hundred Google embedding calls. It is user-initiated and matches the spec's run command verbatim.

**Files:**
- Output: `tamu_data/evals/golden_sets/ragas_20260518.xlsx` (date will match run date)

- [ ] **Step 1: Confirm Phase 1 artifacts and personas are in place**

Run:

```bash
ls -la /workspace/tamu_data/evals/ragas_kg.json /workspace/tamu_data/evals/personas/course_shopping_student.yaml
```

Expected: both files exist; the KG cache is ~7.4 MB.

- [ ] **Step 2: Run Phase 2 generation**

Run:

```bash
cd /workspace && python -m tamubot.evals.generate_ragas_testset \
  --corpus-dir /workspace/data/syllabi/silver/06_chunk \
  --include-terms 202611,202621,202641 \
  --provider tamu \
  --distribution balanced_50_50 \
  --target-size 20 \
  --batch-size 5 \
  --persona-file tamu_data/evals/personas/course_shopping_student.yaml \
  2>&1 | tee /tmp/ragas_phase2.log
```

Expected: the script loops up to ~4 batches, prints `Loaded 1 persona(s): ['Course-Shopping Student']`, and lands `tamu_data/evals/golden_sets/ragas_YYYYMMDD.xlsx` with 20 rows. Watch for `TAMU` gateway errors — the workaround should suppress them; surface anything that isn't a quietly retried SSE chunk.

- [ ] **Step 3: Verify acceptance criteria 1, 3, 4 programmatically**

Run:

```bash
python <<'PY'
import json, pathlib, openpyxl
xlsx_dir = pathlib.Path("/workspace/tamu_data/evals/golden_sets")
latest = sorted(xlsx_dir.glob("ragas_*.xlsx"))[-1]
print("Inspecting:", latest)
wb = openpyxl.load_workbook(latest)
ws = wb.active
header = [c.value for c in next(ws.iter_rows(max_row=1))]
rows = list(ws.iter_rows(min_row=2, values_only=True))
counts = {}
crn_set = 0
crn_blank_hybrid = 0
n_hybrid = 0
for r in rows:
    row = dict(zip(header, r))
    fn = row.get("expected_function") or ""
    counts[fn] = counts.get(fn, 0) + 1
    if fn == "hybrid_course":
        n_hybrid += 1
        if not (row.get("crn") or "").strip():
            crn_blank_hybrid += 1
    if (row.get("crn") or "").strip():
        crn_set += 1
print(f"Rows: {len(rows)}")
print(f"expected_function counts: {counts}")
print(f"CRN populated: {crn_set}/{len(rows)}")
print(f"hybrid_course rows with blank CRN: {crn_blank_hybrid}/{n_hybrid}")
PY
```

Expected:
- `Rows: 20`
- `expected_function counts: {'hybrid_course': ~10, 'semantic_general': ~10}` (±2)
- For hybrid_course rows, CRN populated for ≥80% (`crn_blank_hybrid / n_hybrid ≤ 0.20`)

Stop and investigate if the distribution is far off or if hybrid_course CRN coverage is under 80%.

- [ ] **Step 4: Manual spot-check for tone and transient-deadline leakage (acceptance 2 + 5)**

Open the XLSX and read every question. For each row, verify:

- **Tone:** Reads as a student deciding whether to take the course next semester, not as a generic third-party summariser or a currently-enrolled student.
- **No transient deadlines:** No mentions of "this week", "next Friday", "what's due now", a specific current-term date (e.g., "September 15"), or "what's on the next quiz".

Record any failing rows. If ≥2 rows fail either check, do not proceed — surface the failures and either tighten `SELECTION_TIME_NUDGE` or refine the persona `role_description`, then rerun a single batch with `--target-size 5` to confirm before regenerating the full set.

- [ ] **Step 5: Update the project memory entry**

This step is performed by the user (memory writes are not part of the implementation diff). Suggested update to `[[project_ragas_kg_isen]]`: mark Phase 2 complete, link to the produced XLSX, note any nudge/persona tweaks discovered during spot-check.

---

## Self-Review

**Spec coverage:**

| Spec section | Implemented by |
|---|---|
| Persona definition | Task 2 (YAML), Task 4 (loader) |
| Single-hop synthesizer consumes `entities` | Unchanged — already the default, called out in Task 6 docstring |
| Multi-hop abstract uses `summary_similarity` | Unchanged — already the default |
| Prompt nudge appended to single-hop only | Task 6 (`_apply_selection_time_nudge`) + Task 5 negative test for multi-hop abstract |
| New `--persona-file` flag with `""` escape hatch | Task 7 step 1 + step 5 smoke test |
| `personas.py` loader, fails loud | Task 3 + Task 4 (`FileNotFoundError`, `ValueError`) |
| Edits to `build_query_distribution`, `generate_testset`, `main` | Tasks 6 and 7 |
| KG cache reused | Task 7 step 6 (`--build-kg-only` proves cache hit) |
| Run command matches spec | Task 8 step 2 verbatim |
| Acceptance criteria 1, 3, 4 | Task 8 step 3 |
| Acceptance criteria 2, 5 (manual) | Task 8 step 4 |

**Placeholder scan:** No `TBD`, `TODO`, "implement later", or "add appropriate handling" anywhere. All code blocks contain the actual code.

**Type consistency:** `load_personas` → `list[Persona]` consistent across Task 3, Task 4, Task 7. `build_query_distribution(llm, preset)` signature unchanged. `_apply_selection_time_nudge` is private to the module and used only internally.

**Risk flagged but unmitigated by this plan (intentionally):** The TAMU SSE workaround is load-bearing. If Task 8 step 2 fails mid-batch with an SSE parsing error, debug `tamu_openai_workaround.py` before suspecting the persona/nudge changes — none of the changes in this plan touch the OpenAI client wrapping.
