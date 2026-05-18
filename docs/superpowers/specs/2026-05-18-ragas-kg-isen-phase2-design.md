# Ragas KG ISEN — Phase 2 QA Generation (Persona-Driven)

**Date:** 2026-05-18
**Status:** Design approved, pending implementation plan
**Predecessor:** Phase 1 KG built and cached at `tamu_data/evals/ragas_kg.json` on 2026-05-16

## Problem

Phase 1 of the Ragas KG ISEN initiative produced a cached knowledge graph
covering 75 ISEN syllabi across Spring/Summer/Fall 2026 (88 document nodes,
122 chunk nodes, 2,481 summary-similarity edges, 572 entities-overlap edges).
The generator script (`src/tamubot/evals/generate_ragas_testset.py`) already
supports a `balanced_50_50` distribution that produces 50% single-hop-specific
items (→ `hybrid_course` retrieval) and 50% multi-hop-abstract items
(→ `semantic_general` retrieval).

A naive Phase 2 run today produces homogeneous questions because Ragas's
default scenario only varies *nodes*; query length, style, and persona fall
back to internal defaults. The result is a 20-item set whose tone reads as
one anonymous student wrote every question, with no bias toward the real
TamuBot use case.

## User and use case

TamuBot's primary user for the ISEN syllabus corpus is a Texas A&M student
**choosing courses for a future semester**. Once enrolled in a course, the
student rarely returns to the bot — operational mid-semester questions
("what's on next week's quiz", "when is the assignment due") are out of
scope for this eval set.

The eval set must therefore exercise two question shapes:

1. **Single-course factual lookups** — content/topics, learning outcomes,
   schedule overview, tools/software/textbooks, grading breakdown, exam
   structure, modality, attendance policy, listed prerequisites.
2. **Cross-course comparisons** — which courses cover a given topic, which
   use a particular tool, how two courses differ in workload or focus.

Personalized recursive questions ("given I took CSCE 411, am I ready for
ISEN 489") are deferred to a future eval set built on
`MultiHopSpecificQuerySynthesizer` → `recursive` retrieval.

## Design

### Persona

A single `Persona` is defined and passed to `TestsetGenerator.generate(...)`:

```yaml
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

Manual definition (not `generate_personas_from_kg`) — the user is
well-characterised and we want the persona to reflect a product decision,
not a sampled-from-KG approximation.

### KG property mapping

The cached KG already carries the properties needed by both synthesizers in
`balanced_50_50`; no KG rebuild is required.

| Synthesizer (weight)                       | Reads                                                | Maps to              |
| ------------------------------------------ | ---------------------------------------------------- | -------------------- |
| `SingleHopSpecificQuerySynthesizer` (0.50) | `entities` on CHUNK nodes (syllabus-tuned NER)       | `hybrid_course`      |
| `MultiHopAbstractQuerySynthesizer` (0.50)  | `summary` on DOCUMENT nodes + `summary_similarity`  | `semantic_general`   |

The syllabus-tuned NER prompt (`_build_syllabus_ner_prompt`) already biases
entity extraction toward selection-time-relevant facts (course IDs,
instructors, grading %, prereqs, textbooks, exam formats, tools, credit
hours, dates). The single-hop synthesizer consumes these directly.

The multi-hop abstract synthesizer walks `summary_similarity` edges between
documents — naturally cross-course, naturally comparative.

### Prompt nudge

The `generate_query_reference_prompt` on
`SingleHopSpecificQuerySynthesizer` is mutated at build time to append:

> *The student is choosing courses for a future semester and is not
> currently enrolled. Prefer questions about durable course attributes —
> topics covered, learning outcomes, tools used, grading structure,
> prerequisites. Avoid questions about specific term-bound deadlines (e.g.,
> "when is the midterm scheduled" or "what is due next week").*

Override pattern (per Ragas docs):

```python
prompt = synth.get_prompts()["generate_query_reference_prompt"]
prompt.instruction = prompt.instruction + "\n\n" + NUDGE
synth.set_prompts(**{"generate_query_reference_prompt": prompt})
```

No prompt change to the multi-hop abstract synthesizer — comparison phrasing
is its default behavior.

## Implementation footprint

### New: `tamu_data/evals/personas/course_shopping_student.yaml`

The persona file above. YAML so non-developers can edit copy without
touching code.

### New: `src/tamubot/evals/personas.py`

Small loader (~25 lines):

```python
def load_personas(path: Path) -> list[Persona]:
    """Load Ragas Personas from a YAML file. Fail loud if missing/empty."""
```

Returns `list[ragas.testset.persona.Persona]`. Raises `FileNotFoundError`
if path doesn't exist; raises `ValueError` if the file has no `personas:`
list or the list is empty.

### Edit: `src/tamubot/evals/generate_ragas_testset.py`

Four targeted changes:

1. **CLI flag** — add `--persona-file` (default
   `tamu_data/evals/personas/course_shopping_student.yaml`; pass
   `--persona-file ""` to skip and fall back to Ragas defaults).
2. **`build_query_distribution(...)`** — when constructing
   `SingleHopSpecificQuerySynthesizer`, apply the prompt nudge via the
   `get_prompts` / `set_prompts` override pattern.
3. **`generate_testset(...)`** — accept `persona_list: list[Persona]` and
   pass to `generator.generate(persona_list=persona_list, ...)`.
4. **`main()`** — when `--persona-file` is non-empty, load the personas
   and thread through. Log persona names + count at startup.

No changes to: KG build, embeddings, validation, exporter, schema. The
cached KG is loaded as-is.

## Run command

```bash
cd /workspace && python -m tamubot.evals.generate_ragas_testset \
  --corpus-dir /workspace/data/syllabi/silver/06_chunk \
  --include-terms 202611,202621,202641 \
  --provider tamu \
  --distribution balanced_50_50 \
  --target-size 20 \
  --batch-size 5 \
  --persona-file tamu_data/evals/personas/course_shopping_student.yaml
```

Output: `tamu_data/evals/golden_sets/ragas_20260518.xlsx` (or whatever date
the run lands on).

## Acceptance criteria

1. **Distribution sanity** — `expected_function` column is approximately
   10 `hybrid_course` / 10 `semantic_general` (±2 due to validation drops).
2. **No transient-deadline questions** — manual spot-check confirms zero
   items mention "this week", "next Friday", specific current-term dates,
   or "what's due now". If any slip through, tighten the nudge wording
   and rerun a batch.
3. **All items grounded** — `reference_contexts` fuzzy-match
   `06_chunk/*.json` `chunks[].content` at ≥0.7 ratio. (Existing
   `validate_testset` already enforces this.)
4. **CRN coverage** — `crn` column is non-empty for ≥80% of single-hop
   items. Multi-hop abstract may legitimately span multiple CRNs and
   leave it blank.
5. **Persona tone visible** — manual read confirms questions read as a
   student deciding which course to take, not as a generic third-party
   summariser.

## Out of scope

- Recursive / personalized questions ("given my background…") — future
  eval set on `MultiHopSpecificQuerySynthesizer`.
- Dual `SingleHopSpecificQuerySynthesizer` split (`headlines` +
  `keyphrases`) — kept as a fallback if the 20-item batch reads samey;
  not implemented up-front.
- `generate_personas_from_kg` auto-derivation — manual definition is
  more honest about who we expect to use TamuBot.
- KG rebuild — cached KG is sufficient.

## Cost and risk

- **LLM cost**: roughly 50–80 chat completions through the TAMU gateway
  for 20 validated items at `batch-size=5` (scenario generation +
  theme/persona matching + Q&A generation + Ragas's internal filter),
  plus a few hundred embedding calls to Google direct for theme/persona
  matching. Within TAMU rate limits observed during Phase 1.
- **Primary risk**: TAMU SSE-as-string + low-max-tokens behaviour is
  load-bearing. If generation fails at batch 1, suspect
  `tamu_openai_workaround.py` before the persona change.
- **Secondary risk**: An over-strict prompt nudge could collapse single-
  hop questions onto one or two entity types (e.g., grading %). Mitigated
  by acceptance criterion 5 — re-tune the nudge if seen.

## References

- Ragas docs: persona system, `generate_personas_from_kg`, custom prompt
  override pattern, `SingleHopSpecificQuerySynthesizer` property selection.
- Phase 1 memory: `project_ragas_kg_isen.md`.
- TAMU gateway quirks: same memory file + `tamu_openai_workaround.py`.
