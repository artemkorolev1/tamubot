# Parallel build comparison — slugify-helper (2026-05-28)

## Task spec

Add a pure helper function `slugify(text: str) -> str` to a NEW module `src/tamubot/core/text_utils.py`. Behavior: lowercase the input; strip leading/trailing whitespace; replace any run of characters that are not ASCII letters or digits with a single hyphen; collapse multiple hyphens into one; strip leading/trailing hyphens. Required examples: slugify('Hello, World!')=='hello-world'; slugify('  CSCE 421 -- Machine Learning  ')=='csce-421-machine-learning'; slugify('')==''; slugify('***')==''. Tests cover these plus edge cases (accented/unicode, digits-only). Dependency-free (stdlib only).

## At a glance

| arm | files changed | tests added | tests pass | lint clean | LOC (+/-) | self-confidence |
|---|---|---|---|---|---|---|
| superpowers | 3 (module, tests, RESULTS.md) | 9 | 9/9 pass (verified) | yes (verified) | +124 total / +72 code+tests | 5 |
| mattpocock | 3 (module, tests, RESULTS.md) | 8 | 8/8 pass (verified) | yes (verified) | +137 total / +81 code+tests | 5 |
| vanilla | 3 (module, tests, RESULTS.md) | 9 | 9/9 pass (verified) | yes (verified) | +116 total / +66 code+tests | 5 |

(LOC "total" includes RESULTS.md; "code+tests" counts only `text_utils.py` + `test_text_utils.py` insertions. All test/lint results independently re-run by the reviewer, not taken from the self-reports.)

## How each methodology shaped the approach

**superpowers** — The RESULTS.md narrates an explicit RED→GREEN cycle (9 failing tests first, confirmed `ModuleNotFoundError`, then minimal implementation) and a verification/adversarial-probe step standing in for `requesting-code-review` (subagents were disallowed in this arm). The visible output is the most heavily verified self-report (isolation probe, import-resolution probe, adversarial cases), but the resulting code is essentially the same three-line core as the other two arms — the methodology shaped the *process narrative* far more than the artifact.

**mattpocock** — RESULTS.md describes a "tracer-bullet" first test (`Hello, World!`) widened incrementally via the `tdd` skill. The artifact has the longest/most explanatory docstring (numbered transformation steps + an explicit non-ASCII note) and inline comments explaining the regex-run rationale. It has the fewest tests (8) but the only arm that asserts *two* examples per edge case (e.g. `Café Olé` and `naïve`; `42` and `007`). The "deep module, tiny surface" framing is visible in the prose, not in materially different code.

**vanilla** — Shortest module (28 lines), no methodology framing. Tests are terse one-assert functions. It is the only arm whose tests include a non-Latin-script case (`日本語`) and an explicit whitespace-only case (`'   \t\n '`). The output is the leanest of the three while still covering the required examples and the spec-named edge cases.

## Independent correctness / risk read (per arm)

I re-ran all three test suites (superpowers 9/9, mattpocock 8/8, vanilla 9/9 — all green), re-ran `ruff check` on each (all "All checks passed!"), confirmed each module's docstring passes `python -m doctest` (4/4 each), and behaviorally probed all three via `PYTHONPATH=<worktree>/src` against an identical battery of inputs. **All three produced byte-identical output on every case**, including the spec's four required examples and additional adversarial inputs (`'  -x-  '` → `'x'`, `'a___b...c'` → `'a-b-c'`, `'İ'` (Turkish dotted capital I) → `'i'`).

The core of all three is the same and is correct against the spec:
```python
lowered = text.strip().lower()            # vanilla, superpowers
# (mattpocock: text.lower().strip())
hyphenated = _RE.sub("-", lowered)        # _RE = re.compile(r"[^a-z0-9]+")
return hyphenated.strip("-")
```
The single regex `[^a-z0-9]+` matches a *maximal run* of non-alnum chars, so "replace separator runs with one hyphen" and "collapse multiple hyphens" are satisfied in one pass — the spec's "collapse multiple hyphens into one" never needs a second `re.sub('-+', ...)` step. The trailing `.strip("-")` cleans up edge hyphens left by `***` / `--abc--`. This is correct.

- **superpowers** — Correct. The precompiled module-level regex is pure with no import side effects. Risk read: none functional. The RESULTS.md claim that pytest imports the worktree module (not the container-wide `/workspace/src` install) is **true and load-bearing** — I verified a bare `python -c "import tamubot.core.text_utils"` fails (`ModuleNotFoundError`) because the installed `tamubot` resolves to `/workspace/src`, while pytest succeeds because `pyproject.toml` sets `pythonpath = [".", "src"]`, putting the worktree's `src` first. So the green tests genuinely exercise this arm's code. Test set is the broadest single-assert coverage (9 tests incl. unicode-only `'éè'` → `''` and internal-run collapse).

- **mattpocock** — Correct. Ordering is `text.lower().strip()` rather than `.strip().lower()`. I checked whether the ordering can ever diverge: Python's `str.lower()` never introduces or removes leading/trailing whitespace (even for `İ`, `ß`, etc.), so `.lower().strip()` and `.strip().lower()` are behaviorally equivalent here — no bug, just a stylistic difference. Slight risk note: this arm has the *fewest* test functions (8) and omits an explicit empty-after-strip-from-whitespace case and a non-Latin-script case that the other two include; coverage is still adequate for the spec because it doubles up assertions inside its edge-case tests. Best-documented module of the three.

- **vanilla** — Correct. Leanest implementation; same `.strip().lower()` ordering as superpowers. Tests are the most spec-literal and add the only CJK case (`'日本語'` → `''`) and the only pure-whitespace case (`'   \t\n '` → `''`). Risk read: none functional. Docstring is doctest-valid despite the report only loosely claiming it.

Cross-cutting risk for all three: the spec says "ASCII letters or digits", and all three interpret this strictly — accented/non-Latin letters become separators rather than being transliterated (`Café déjà vu` → `caf-d-j-vu`). All three RESULTS.md explicitly flag this as a deliberate reading and pin it with a test. This matches the literal spec wording, but a downstream caller expecting `café` → `cafe` would be surprised; that is a spec-interpretation risk shared equally, not a bug in any one arm.

## Notable divergences worth the research agent's attention

- **strip/lower ordering**: mattpocock uses `.lower().strip()`; superpowers and vanilla use `.strip().lower()`. Verified behaviorally identical for all inputs — flag only as a stylistic/consistency observation, not a correctness difference.
- **Test count & style**: superpowers (9) and vanilla (9) use one-assert-per-function; mattpocock (8) packs two assertions into several edge-case tests. Net assertion coverage is comparable; granularity (and thus failure localization) is finer in superpowers/vanilla.
- **Edge-case coverage divergence**: only vanilla tests a non-Latin script (`日本語`) and an explicit multi-char whitespace-only input; only superpowers tests unicode-*only* (`éè` → `''`) as a distinct case and an explicit internal-run collapse (`a   b___c...d`); mattpocock uniquely tests two accented words and underscore/tab/newline mixing in one assertion. No single arm is a strict superset of the others' edge cases.
- **Docstring depth**: mattpocock's docstring is the most thorough (numbered steps + explicit non-ASCII note); vanilla's is the shortest; all three are doctest-valid and pass `python -m doctest`. Two arms (superpowers, vanilla) advertise wanting doctest-in-CI as future work.
- **Comments in the module**: mattpocock and superpowers name the regex with an explanatory comment (`_SEPARATOR_RUN` / `_NON_ALNUM_RUN` with a comment); vanilla uses `_NON_ALNUM_RUN` with no comment. Naming convergence on `_NON_ALNUM_RUN` between superpowers and vanilla.
- **Self-report verification rigor**: superpowers documents the most verification (import-resolution probe, isolation check, adversarial probe); mattpocock and vanilla report standard pytest+ruff runs. The reviewer independently confirmed all reported pass/lint claims are accurate for every arm.

## Branches & worktree paths

- superpowers — branch `pb/slugify-helper-superpowers` — worktree `/workspace/.claude/worktrees/pb-slugify-helper-superpowers`
- mattpocock — branch `pb/slugify-helper-mattpocock` — worktree `/workspace/.claude/worktrees/pb-slugify-helper-mattpocock`
- vanilla — branch `pb/slugify-helper-vanilla` — worktree `/workspace/.claude/worktrees/pb-slugify-helper-vanilla`
