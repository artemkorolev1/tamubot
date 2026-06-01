# TASK

Fix issue {{TASK_ID}}: {{ISSUE_TITLE}}

Pull in the issue using `gh issue view <ID>`. If it has a parent PRD, pull that in too.

Only work on the issue specified.

Work on branch {{BRANCH}}. Make commits and run tests.

# CONTEXT

Here are the last 10 commits:

<recent-commits>

!`git log -n 10 --format="%H%n%ad%n%B---" --date=short`

</recent-commits>

# EXPLORATION

Explore the repo and fill your context window with relevant information that will allow you to complete the task.

Pay extra attention to test files that touch the relevant parts of the code.

# EXECUTION

If applicable, use RGR to complete the task.

1. RED: write one test
2. GREEN: write the implementation to pass that test
3. REPEAT until done
4. REFACTOR the code

# FEEDBACK LOOPS

Before committing, verify your change with the project's `make` targets:

1. `make lint` (ruff) — **must pass**
2. `make typecheck` (mypy with `ignore_missing_imports`) — **must pass**
3. `make test` (pytest) — run it; some tests import heavy/GPU deps (torch, transformers, bitsandbytes, flash-linear-attention, opendataloader-pdf, dagster, docling-slim) that are intentionally absent from this sandbox. If `make test` reports `ModuleNotFoundError` for one of those, note it in the commit message under "Blockers or notes for next iteration" and proceed on the strength of lint + typecheck. If a test for code you actually changed fails, fix it first.

# COMMIT

Make a git commit. The commit message must:

1. Start with `RALPH:` prefix
2. Include task completed + PRD reference
3. Key decisions made
4. Files changed
5. Blockers or notes for next iteration

Keep it concise.

# THE ISSUE

If the task is not complete, leave a comment on the issue with what was done.

Do not close the issue - this will be done later.

Once complete, output <promise>COMPLETE</promise>.

# FINAL RULES

ONLY WORK ON A SINGLE TASK.
