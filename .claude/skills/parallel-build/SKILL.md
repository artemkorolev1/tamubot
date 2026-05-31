---
name: parallel-build
description: Use when building a feature three ways in parallel to compare methodologies — builds the same spec in superpowers / Matt Pocock / vanilla arms in isolated git worktrees, then emits a comparison.md for downstream research-agent analysis. Trigger on requests like "build this three ways", "parallel-build", "methodology bake-off".
---

# parallel-build

Build ONE feature spec three times, one arm per methodology, then synthesize a comparison
file. No arm is auto-merged; the operator/research agent decides.

## CRITICAL SAFETY (learned from a real incident)

A prior run let arms run `git add -A && git commit` in the MAIN worktree, which swept
~700 uncommitted files into accidental commits on `main`. The design below makes that
structurally impossible. NON-NEGOTIABLE rules:

1. The **orchestrator pre-creates** each worktree with an explicit path + branch. Do NOT
   rely on the `Agent` tool's `isolation: worktree` for multiple arms — parallel
   auto-provisioning proved unreliable (only 1 of 3 got a worktree).
2. Every arm does ALL git via `git -C <its-worktree-path> ...`. Arms NEVER run git in
   `/workspace`, NEVER `cd /workspace`, NEVER use `git add -A`.
3. Arms stage ONLY the specific files they create, by name.
4. Arms write files ONLY under their worktree path, never to `/workspace/...`.
5. Dispatch arms **sequentially** (one `Agent` call, wait, next). Parallel can be
   reconsidered later, but sequential matches the one mechanism that worked reliably.

## Preconditions (check, then proceed)

1. A clear feature spec/task. If vague, ask before dispatching.
2. Warn the operator: arms branch from current committed HEAD — uncommitted working-tree
   changes are NOT visible. Ask them to commit/stash dependencies first if relevant.
3. Context7 is allowed (`mcp__plugin_context7_context7__*` in `.claude/settings.local.json`).
4. git trusts the repo: `~/.config/git/config` has `safe.directory = /workspace`
   (else `git worktree add` fails with "dubious ownership").

## Procedure

### 1. Record base + slug
```bash
git rev-parse HEAD   # = BASE
```
Pick a kebab `<slug>` and today's `<YYYY-MM-DD>`.

### 2. Pre-create the three worktrees (orchestrator, deterministic)
```bash
for arm in superpowers mattpocock vanilla; do
  git worktree add "/workspace/.claude/worktrees/pb-<slug>-$arm" -b "pb/<slug>-$arm" <BASE>
done
git worktree list   # confirm all three exist on their pb/<slug>-* branches
```

### 3. Dispatch arms ONE AT A TIME (sequential)
For each arm, make a single `Agent` call (`subagent_type: general-purpose`, NO `isolation`
param), wait for it to finish, then do the next. Use the ARM PROMPT TEMPLATE, substituting
`<arm>`, `<WORKTREE>` (its path), `<BRANCH>` (`pb/<slug>-<arm>`), the verbatim spec, `<BASE>`,
and `<slug>`.

Arm methodology directives:
- **superpowers** — "Follow the superpowers methodology: invoke superpowers:test-driven-development (failing tests first), superpowers:verification-before-completion, superpowers:requesting-code-review. You CANNOT spawn subagents — do NOT use subagent-driven-development."
- **mattpocock** — "Follow the Matt Pocock methodology. Use as relevant: tdd, diagnose, zoom-out, improve-codebase-architecture. Do NOT use superpowers:* skills."
- **vanilla** — "Do NOT invoke ANY process/workflow skill. Implement directly. You MAY use Context7 for library docs."

### 4. Collect results
Each arm returns: confirmed branch, worktree path, summary, test+lint status. Record the
three `(arm, branch, worktree_path)` tuples.

### 5. Dispatch the evaluator (ONE Agent call, NO isolation, runs in /workspace)
`subagent_type: general-purpose`, using the EVALUATOR PROMPT. It reads each arm via its
worktree path, writes `comparison.md`. It must NOT modify code and must NOT pick a winner.

### 6. Assemble + report
```bash
OUT="docs/parallel-builds/<YYYY-MM-DD>-<slug>"
mkdir -p "$OUT"
cp "/workspace/.claude/worktrees/pb-<slug>-superpowers/RESULTS.md" "$OUT/superpowers.md"
cp "/workspace/.claude/worktrees/pb-<slug>-mattpocock/RESULTS.md"  "$OUT/mattpocock.md"
cp "/workspace/.claude/worktrees/pb-<slug>-vanilla/RESULTS.md"     "$OUT/vanilla.md"
# evaluator already wrote "$OUT/comparison.md"
```
Print the three branches, the three worktree paths, and the path to `comparison.md`.
State that nothing was merged.

### 7. Cleanup — ASK FIRST
Never remove worktrees/branches without explicit operator confirmation. Offer:
```bash
git worktree remove "/workspace/.claude/worktrees/pb-<slug>-<arm>" --force
git branch -D "pb/<slug>-<arm>"
```

---

## ARM PROMPT TEMPLATE

```
You are ONE arm of a three-way parallel build. Your ONLY workspace is the git worktree at:
    <WORKTREE>
which is checked out on branch <BRANCH>, branched from BASE=<BASE>.

=== HARD SAFETY RULES (violating these corrupts the main repo) ===
- Run ALL git commands as `git -C <WORKTREE> ...`. NEVER run git anywhere else.
- NEVER `cd /workspace`. NEVER write or edit any file under /workspace that is NOT under
  <WORKTREE>. All file paths you Write/Edit MUST start with <WORKTREE>/.
- NEVER use `git add -A` or `git add .`. Stage ONLY the specific files you create, by name.
- FIRST ACTION: run `git -C <WORKTREE> branch --show-current`. If it does NOT print exactly
  <BRANCH>, STOP immediately, write nothing, commit nothing, and report "NOT ISOLATED".

FEATURE SPEC (build exactly this, with files under <WORKTREE>/):
<verbatim feature spec>

METHODOLOGY (soft rule for THIS arm):
<methodology directive for this arm>

DEFINITION OF DONE:
1. Implement the feature (files under <WORKTREE>/ only).
2. Run `ruff check --fix` and `ruff format` on ONLY the files you created.
3. Run ONLY the targeted tests for this feature with pytest (NOT the full `make test`). Record pass/fail.
4. Write <WORKTREE>/RESULTS.md (schema below).
5. `git -C <WORKTREE> add <each file you created>` then
   `git -C <WORKTREE> commit -m "parallel-build <arm>: <slug>"`.

RETURN: confirmed branch (`git -C <WORKTREE> branch --show-current`), the worktree path,
a one-paragraph summary, and test+lint status.

RESULTS.md SCHEMA (exact headings):
# <arm> — <slug>
## Approach summary
## Files changed
- path — one-line rationale
## How I verified
## Key design decisions & tradeoffs
## Difficulties / dead-ends
## Self-assessed confidence (1-5) + what I'd do with more time
```

## EVALUATOR PROMPT

```
You are a skeptical, independent reviewer. You did NOT write any implementation below.
Do not modify any code. Do not declare a winner.

Inputs:
- Feature spec: <verbatim spec>
- BASE commit: <BASE>
- Arms (superpowers / mattpocock / vanilla), each branch + worktree path:
  <arm> -> branch <branch>, worktree <path>

For each arm: read <path>/RESULTS.md and the diff `git -C <path> diff <BASE> -- .`.

Write `docs/parallel-builds/<YYYY-MM-DD>-<slug>/comparison.md` with EXACTLY:
# Parallel build comparison — <slug> (<YYYY-MM-DD>)
## Task spec
## At a glance
| arm | files changed | tests added | tests pass | lint clean | LOC (+/-) | self-confidence |
## How each methodology shaped the approach
## Independent correctness / risk read (per arm)
## Notable divergences worth the research agent's attention
## Branches & worktree paths
Do NOT pick a winner. End the file there.
```
