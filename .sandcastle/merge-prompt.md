# TASK

Merge the following branches into the current branch:

{{BRANCHES}}

For each branch:

1. Run `git merge --no-ff --no-edit <branch>` (the `--no-ff` flag guarantees every merge produces a visible merge commit, so the operator can revert a single branch later with `git revert -m 1 <sha>`).
2. If there are merge conflicts, resolve them intelligently by reading both sides and choosing the correct resolution.
3. After each merge, run `make lint` and `make typecheck` to verify the integrated state is sane. `make test` may surface `ModuleNotFoundError` for heavy/GPU deps absent from this sandbox — note any such failures but do not block the merge on them. Block only on lint, typecheck, or tests for code that actually changed.
4. If lint or typecheck fails, fix the issue before proceeding to the next branch.

The per-branch `--no-ff` commits replace any need for a final "summary commit" — do NOT create one.

# CLOSE ISSUES

For each branch that was merged, close its issue using the following command:

`gh issue close <ID> --comment "Completed by Sandcastle"`

Here are all the issues:

{{ISSUES}}

Once you've merged everything you can, output <promise>COMPLETE</promise>.
