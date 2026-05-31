# mattpocock — slugify-helper

## Approach summary
Built `slugify` with a Matt-Pocock-style TDD loop driven by the `tdd` skill:
one tracer-bullet test (`Hello, World!`) → minimal module → RED→GREEN, then
incrementally widened coverage to the four required examples plus edge cases
(accented/unicode, digits-only, leading/trailing separators, mixed
whitespace/underscores). The implementation is a deep-but-tiny function: a
single precompiled regex `[^a-z0-9]+` does all the separator-collapsing work,
so multi-hyphen collapse falls out for free rather than being a separate pass.
Standard library only (`re`), no new dependencies.

## Files changed
- src/tamubot/core/text_utils.py — new dependency-free module exposing `slugify(text) -> str`.
- tests/test_text_utils.py — unit tests for the four required examples + 4 edge-case tests.
- RESULTS.md — this report.

## How I verified
- `python -m pytest tests/test_text_utils.py -v` → 8 passed, 0 failed.
- Confirmed RED first: initial run failed with `ModuleNotFoundError` before the module existed.
- `ruff check --fix` on both source files → "All checks passed!".
- `ruff format` on both source files → "2 files left unchanged".
- Re-ran pytest after lint/format → still 8 passed.

## Key design decisions & tradeoffs
- **Single regex for everything.** `[^a-z0-9]+` matches *runs* of non-alphanumerics,
  so replacing each run with one hyphen simultaneously satisfies "replace
  separator runs" and "collapse multiple hyphens" — no second collapse step,
  no `re.sub('-+', ...)` follow-up. Fewer moving parts, fewer bugs.
- **Lowercase before matching.** Lowercasing first lets the character class stay
  `a-z` only (no `A-Z`), keeping the alphabet definition in one place.
- **Strip whitespace then strip hyphens.** `.strip()` handles the spec's literal
  whitespace requirement; the final `.strip("-")` removes hyphens the regex
  introduced at the edges (e.g. from `***` or `--Hello--`). `''` and `'***'`
  both correctly yield `''`.
- **Non-ASCII letters are separators, not transliterated.** Spec says "ASCII
  letters or digits", so `Café` → `caf` (accented `é` is dropped). I chose this
  deliberately over adding `unicodedata` transliteration, to honor the
  "ASCII-only alphabet" and "standard library, dependency-free" constraints.
  Documented in the docstring and pinned by a test so the behavior is intentional.
- **Module placement.** Lives in `src/tamubot/core/text_utils.py` next to
  `config.py`; importing it pulls in nothing heavy, so it's safe to use from
  ingestion/RAG code paths.

## Difficulties / dead-ends
None significant. The only judgment call was unicode handling — I confirmed via
a pinned test (`Café Olé` → `caf-ol`) that the ASCII-only interpretation is
what the spec's "ASCII letters or digits" wording implies, rather than guessing
at transliteration.

## Self-assessed confidence (1-5) + what I'd do with more time
**5.** The function is tiny, pure, fully covered by passing tests, lint/format
clean, and every required example is asserted directly. With more time I'd add
a property-based test (e.g. Hypothesis) asserting the output always matches
`^[a-z0-9]+(-[a-z0-9]+)*$` or is empty — but that would add a dependency the
spec forbids, so it stays out of this arm.
