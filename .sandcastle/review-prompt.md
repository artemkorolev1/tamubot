# TASK

Review the work the implementer did for issue {{TASK_ID}}: {{ISSUE_TITLE}} on branch {{BRANCH}}.

Pull in the issue with `gh issue view {{TASK_ID}}`.

# CONTEXT

You are on branch `{{BRANCH}}`. Inspect what the implementer changed:

<diff-against-launch>

!`git log --format="%H%n%ad%n%s%n---" --date=short -n 5`

</diff-against-launch>

<full-diff>

!`git diff $(git merge-base HEAD @{u} 2>/dev/null || git rev-list --max-parents=0 HEAD | tail -1)..HEAD`

</full-diff>

If the diff command above produced nothing useful, fall back to `git show HEAD --stat` and `git show HEAD` to see the latest commit on this branch.

# REVIEW

Read the diff and answer these questions in your head:

1. Does the change actually address what issue {{TASK_ID}} asked for?
2. Does it touch anything *outside* the requested scope?
3. Are there obvious bugs, missing edge cases, or typos?
4. Is the commit message useful (RALPH: prefix, key decisions, files changed)?

# OUTPUT

Post a single concise review comment on the source issue using `gh`:

```
gh issue comment {{TASK_ID}} --body "..."
```

The body must:

- Start with a one-line verdict: `**LGTM**`, `**LGTM with notes**`, or `**Changes requested**`.
- Then up to 5 bullet points of findings (only include bullets if you have something concrete to say — empty bullet sections are noise).
- Then a final line: `Reviewed by Sandcastle reviewer agent.`

# RULES

- DO NOT modify code. DO NOT make new commits. DO NOT push.
- DO NOT close the issue.
- The merger phase runs after you regardless of verdict — your job is to leave a written record, not to block.
- If you genuinely cannot find the changes (empty diff, wrong branch), comment that fact and exit.

Once the comment is posted, output `<promise>COMPLETE</promise>`.
