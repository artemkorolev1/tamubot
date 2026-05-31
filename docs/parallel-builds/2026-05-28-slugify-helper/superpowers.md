# superpowers — slugify-helper

## Approach summary
Built `slugify(text: str) -> str` test-first using the superpowers methodology
(test-driven-development → verification-before-completion → requesting-code-review).
I wrote 9 failing unit tests first (the 4 required spec examples plus edge cases:
accented/unicode chars, unicode-only, digits-only, leading/trailing separators,
internal-run collapsing), confirmed they failed with `ModuleNotFoundError` (RED),
then wrote the minimal implementation to make them pass (GREEN). The helper is a
single-pass regex substitution: lowercase + strip, replace every run of
non-`[a-z0-9]` characters with one hyphen, then strip edge hyphens.

## Files changed
- src/tamubot/core/text_utils.py — new dependency-free module exposing `slugify`; stdlib `re` only.
- tests/test_text_utils.py — 9 unit tests (4 required examples + accented/unicode/digits/edge cases).
- RESULTS.md — this report.

## How I verified
- RED: ran the tests before writing code → failed with `No module named 'tamubot.core.text_utils'` (expected, feature-missing failure).
- GREEN: ran `python -m pytest .../tests/test_text_utils.py -v` → **9 passed**.
- Confirmed the test imports MY worktree module (not the main repo): a probe printed
  `RESOLVED: /workspace/.claude/worktrees/pb-slugify-helper-superpowers/src/tamubot/core/text_utils.py`,
  and the main-repo interpreter raises `ModuleNotFoundError` for `text_utils` (it exists only in my worktree).
- Lint: `ruff check --fix` → "All checks passed!". Format: `ruff format` → unchanged; `ruff format --check` → "2 files already formatted".
- Re-ran all 4 required spec examples directly → all match.
- Adversarial probe (whitespace tabs/newlines, emoji, only-hyphens, underscores, single char, already-a-slug) → all 9 correct.
- Isolation: `git status --short` shows only the two new source files under the worktree.

## Key design decisions & tradeoffs
- **Single regex `[^a-z0-9]+`** does double duty: it both "replaces runs of non-alphanumeric
  with a single hyphen" and "collapses multiple hyphens", because literal input hyphens are
  themselves non-alphanumeric and get absorbed into the matched run. No separate hyphen-collapse pass needed.
- **Lowercase before matching** lets the character class stay `a-z` (no `A-Z`), keeping the pattern minimal.
- **ASCII-only by design**: accented/unicode letters are treated as separators (e.g. `"Café déjà vu"` → `"caf-d-j-vu"`),
  matching the literal spec ("ASCII letters or digits"). I did NOT add unicode transliteration (e.g. `unicodedata`/`unidecode`)
  because the spec says ASCII-only and forbids non-stdlib deps; transliteration would change documented behavior.
- **Module-level precompiled regex** for efficiency on repeated calls; pure function, no import side effects.

## Difficulties / dead-ends
- The package `tamubot` is installed container-wide resolving to `/workspace/src`, not my worktree's `src`.
  I verified (via an in-pytest probe) that pytest's rootdir/path injection causes the test to import MY
  worktree's `text_utils`, and proved the main-repo `tamubot.core` has no `text_utils` module — so the
  passing tests genuinely exercise my code. No edits were made outside the worktree.
- Could not run `superpowers:requesting-code-review` via a subagent (subagents forbidden in this arm),
  so I performed an equivalent rigorous self-review against the diff + an adversarial edge-case probe instead.

## Self-assessed confidence (1-5) + what I'd do with more time
**5.** The function is small, pure, fully covered, lint/format-clean, and matches every documented example
plus extra edge cases. With more time I would: add a `doctest` run to CI so the docstring examples are
executable contracts, consider a `max_length`/truncation option and an optional unicode-transliteration mode
(behind a flag, separate from this stdlib-only helper), and add a property-based (Hypothesis) test asserting
the output always matches `^[a-z0-9]+(-[a-z0-9]+)*$|^$`.
