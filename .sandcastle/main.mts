// Parallel Planner — three-phase orchestration loop
//
// This template drives a multi-phase workflow:
//   Phase 1 (Plan):    An opus agent analyzes open issues, builds a dependency
//                      graph, and outputs a <plan> JSON listing unblocked issues
//                      with their target branch names.
//   Phase 2 (Execute): N sonnet agents run in parallel via Promise.allSettled,
//                      each working a single issue on its own branch.
//   Phase 3 (Merge):   A sonnet agent merges all branches that produced commits.
//
// The outer loop repeats up to MAX_ITERATIONS times so that newly unblocked
// issues are picked up after each round of merges.
//
// Usage:
//   npx tsx .sandcastle/main.mts
// Or add to package.json:
//   "scripts": { "sandcastle": "npx tsx .sandcastle/main.mts" }

import * as sandcastle from "@ai-hero/sandcastle";
import { docker } from "@ai-hero/sandcastle/sandboxes/docker";
import { z } from "zod";
import { execSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Required env (NOT auto-loaded from .sandcastle/.env — invoke with
// `node --env-file=.sandcastle/.env --import tsx .sandcastle/main.mts`
// or export the vars first).
const oauthToken = process.env.CLAUDE_CODE_OAUTH_TOKEN;
const ghToken = process.env.GH_TOKEN;
if (!oauthToken) throw new Error("CLAUDE_CODE_OAUTH_TOKEN is required (generate with `claude setup-token`).");
if (!ghToken) throw new Error("GH_TOKEN is required (GitHub PAT with repo Issues read/write).");
if (process.env.ANTHROPIC_API_KEY) {
  throw new Error("ANTHROPIC_API_KEY is set — it would override Max-subscription auth. Unset it before running.");
}

// Pre-flight: refuse to run on a dirty working tree. The orchestrator's
// per-issue worktrees branch off HEAD; uncommitted changes on the host
// would either be silently invisible to the agents (best case) or get
// captured into a commit by something else mid-run (the cross-stream
// interference we hit in the first smoke). Override with
// SANDCASTLE_ALLOW_DIRTY=1 when you knowingly want to proceed.
const dirty = execSync("git status --porcelain", { encoding: "utf8" }).trim();
if (dirty && process.env.SANDCASTLE_ALLOW_DIRTY !== "1") {
  throw new Error(
    `Working tree is dirty — commit, stash, or set SANDCASTLE_ALLOW_DIRTY=1.\n\n${dirty}\n`,
  );
}
const launchBranch = execSync("git branch --show-current", { encoding: "utf8" }).trim();
console.log(`Launch branch: ${launchBranch}`);

// Shared agent + sandbox: tokens forwarded into every container exec.
const agent = sandcastle.claudeCode("claude-opus-4-7");
const sandbox = docker({
  imageName: "tamubot-sandcastle:local",
  env: { CLAUDE_CODE_OAUTH_TOKEN: oauthToken, GH_TOKEN: ghToken },
});

// The planner emits its plan as JSON inside <plan> tags; Output.object extracts
// and validates it against this schema. We use Zod here, but any Standard
// Schema validator works just as well — Valibot, ArkType, etc. See
// https://standardschema.dev.
const planSchema = z.object({
  issues: z.array(
    z.object({ id: z.string(), title: z.string(), branch: z.string() }),
  ),
});

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Maximum number of plan→execute→merge cycles before stopping.
// Raise this if your backlog is large; lower it for a quick smoke-test run.
const MAX_ITERATIONS = 10;

// Hard cap on total wall-clock runtime. Sandcastle bills against the Max
// subscription quota — without a stop, a thrashing implementer could chew
// through a 5-hour window. Override with SANDCASTLE_MAX_RUNTIME_MS=...
const MAX_RUNTIME_MS = Number(process.env.SANDCASTLE_MAX_RUNTIME_MS ?? 90 * 60 * 1000);

// Implementer iteration cap. The scaffold defaulted to 100, which is wildly
// generous; most well-scoped issues converge in <10 turns. A misbehaving
// agent that loops forever costs real budget — keep this tight unless an
// issue truly needs it.
const IMPLEMENTER_MAX_ITERATIONS = Number(process.env.SANDCASTLE_IMPLEMENTER_MAX_ITERATIONS ?? 30);

// Per-run artifact summary directory.
const runStartedAt = new Date().toISOString().replace(/[:.]/g, "-");
const runsDir = join(".sandcastle", "logs", "runs");
mkdirSync(runsDir, { recursive: true });
const summaryPath = join(runsDir, `${runStartedAt}.md`);
const iterationSummaries: string[] = [
  `# Sandcastle run — ${runStartedAt}`,
  ``,
  `- Launch branch: \`${launchBranch}\``,
  `- Max iterations: ${MAX_ITERATIONS}`,
  `- Implementer max iterations: ${IMPLEMENTER_MAX_ITERATIONS}`,
  `- Hard runtime cap: ${(MAX_RUNTIME_MS / 60_000).toFixed(0)} min`,
  ``,
];
const runStartTime = Date.now();
const writeSummary = () => writeFileSync(summaryPath, iterationSummaries.join("\n") + "\n");
writeSummary();

// Lean setup: no npm install — the default agent prompts call `npm run
// typecheck` / `npm run test`, which package.json maps to `echo ok` no-ops
// for this smoke configuration. No host node_modules need to reach the
// sandbox.
const hooks = {
  sandbox: { onSandboxReady: [{ command: "echo sandbox ready" }] },
};

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
  if (Date.now() - runStartTime > MAX_RUNTIME_MS) {
    const note = `\nRuntime cap of ${(MAX_RUNTIME_MS / 60_000).toFixed(0)} min exceeded — stopping before iteration ${iteration}.\n`;
    console.log(note);
    iterationSummaries.push(note);
    writeSummary();
    break;
  }
  console.log(`\n=== Iteration ${iteration}/${MAX_ITERATIONS} ===\n`);
  const iterStart = Date.now();
  const iterLog: string[] = [`## Iteration ${iteration}`, ``];

  // -------------------------------------------------------------------------
  // Phase 1: Plan
  //
  // The planning agent (opus, for deeper reasoning) reads the open issue list,
  // builds a dependency graph, and selects the issues that can be worked in
  // parallel right now (i.e., no blocking dependencies on other open issues).
  //
  // It outputs a <plan> JSON block — Output.object parses and validates it.
  // -------------------------------------------------------------------------
  const plan = await sandcastle.run({
    hooks,
    sandbox,
    name: "planner",
    // One iteration is enough: the planner just needs to read and reason,
    // not write code. (Structured output requires maxIterations: 1.)
    maxIterations: 1,
    agent,
    promptFile: "./.sandcastle/plan-prompt.md",
    // Extract and validate the <plan> JSON into a typed object. Throws
    // StructuredOutputError if the tag is missing, the JSON is malformed, or
    // validation fails — which aborts the loop.
    output: sandcastle.Output.object({ tag: "plan", schema: planSchema }),
  });

  const issues = plan.output.issues;

  if (issues.length === 0) {
    // No unblocked work — either everything is done or everything is blocked.
    console.log("No unblocked issues to work on. Exiting.");
    iterLog.push(`Planner found 0 unblocked issues — loop exited.`, ``);
    iterationSummaries.push(...iterLog);
    writeSummary();
    break;
  }

  console.log(
    `Planning complete. ${issues.length} issue(s) to work in parallel:`,
  );
  iterLog.push(`### Planner`, ``, `Selected ${issues.length} issue(s):`, ``);
  for (const issue of issues) {
    console.log(`  ${issue.id}: ${issue.title} → ${issue.branch}`);
    iterLog.push(`- #${issue.id} "${issue.title}" → \`${issue.branch}\``);
  }
  iterLog.push(``);

  // -------------------------------------------------------------------------
  // Phase 2: Execute
  //
  // Spawn one sonnet agent per issue, all running concurrently.
  // Each agent works on its own branch so there are no conflicts during
  // execution — merging happens in Phase 3.
  //
  // Promise.allSettled means one failing agent doesn't cancel the others.
  // -------------------------------------------------------------------------
  const settled = await Promise.allSettled(
    issues.map((issue) =>
      sandcastle.run({
        hooks,
        // Each agent starts on its own branch via branchStrategy on run().
        sandbox,
        branchStrategy: { type: "branch", branch: issue.branch },
        name: "implementer",
        maxIterations: IMPLEMENTER_MAX_ITERATIONS,
        agent,
        promptFile: "./.sandcastle/implement-prompt.md",
        // Prompt arguments substitute {{TASK_ID}}, {{ISSUE_TITLE}},
        // and {{BRANCH}} placeholders in implement-prompt.md before the
        // agent sees the prompt.
        promptArgs: {
          TASK_ID: issue.id,
          ISSUE_TITLE: issue.title,
          BRANCH: issue.branch,
        },
      }),
    ),
  );

  // Log any agents that threw (network error, sandbox crash, etc.).
  for (const [i, outcome] of settled.entries()) {
    if (outcome.status === "rejected") {
      console.error(
        `  ✗ ${issues[i]!.id} (${issues[i]!.branch}) failed: ${outcome.reason}`,
      );
    }
  }

  // Only pass branches that actually produced commits to the merge phase.
  // An agent that ran successfully but made no commits has nothing to merge.
  const completedIssues = settled
    .map((outcome, i) => ({ outcome, issue: issues[i]! }))
    .filter(
      (
        entry,
      ): entry is {
        outcome: PromiseFulfilledResult<
          Awaited<ReturnType<typeof sandcastle.run>>
        >;
        issue: (typeof issues)[number];
      } =>
        entry.outcome.status === "fulfilled" &&
        entry.outcome.value.commits.length > 0,
    )
    .map((entry) => entry.issue);

  const completedBranches = completedIssues.map((i) => i.branch);

  console.log(
    `\nExecution complete. ${completedBranches.length} branch(es) with commits:`,
  );
  iterLog.push(`### Implementers`, ``);
  for (const [i, outcome] of settled.entries()) {
    const issue = issues[i]!;
    if (outcome.status === "rejected") {
      iterLog.push(`- #${issue.id} (${issue.branch}) — **failed**: \`${String(outcome.reason).slice(0, 200)}\``);
    } else {
      const nCommits = outcome.value.commits.length;
      iterLog.push(`- #${issue.id} (${issue.branch}) — ${nCommits} commit(s)`);
    }
  }
  iterLog.push(``);
  for (const branch of completedBranches) {
    console.log(`  ${branch}`);
  }

  if (completedBranches.length === 0) {
    // All agents ran but none made commits — nothing to merge this cycle.
    console.log("No commits produced. Nothing to merge.");
    iterLog.push(`No commits produced this iteration; skipping review + merge.`, ``);
    iterationSummaries.push(...iterLog);
    writeSummary();
    continue;
  }

  // -------------------------------------------------------------------------
  // Phase 3: Review
  //
  // One reviewer agent per completed branch, in parallel. Each reviewer
  // checks out the implementer's branch, diffs against the launch branch,
  // and posts a brief code-review comment on the source issue via gh.
  // Review is advisory — findings appear as issue comments; the merge
  // proceeds regardless. (Tighten later by filtering on review verdict.)
  // -------------------------------------------------------------------------
  const reviewSettled = await Promise.allSettled(
    completedIssues.map((issue) =>
      sandcastle.run({
        hooks,
        sandbox,
        branchStrategy: { type: "branch", branch: issue.branch },
        name: "reviewer",
        maxIterations: 20,
        agent,
        promptFile: "./.sandcastle/review-prompt.md",
        promptArgs: {
          TASK_ID: issue.id,
          ISSUE_TITLE: issue.title,
          BRANCH: issue.branch,
        },
      }),
    ),
  );

  iterLog.push(`### Reviewers`, ``);
  for (const [i, outcome] of reviewSettled.entries()) {
    const issue = completedIssues[i]!;
    if (outcome.status === "rejected") {
      console.error(
        `  ✗ review of ${issue.id} (${issue.branch}) failed: ${outcome.reason}`,
      );
      iterLog.push(`- #${issue.id} — **failed**: \`${String(outcome.reason).slice(0, 200)}\``);
    } else {
      iterLog.push(`- #${issue.id} — completed (see issue comment for verdict)`);
    }
  }
  iterLog.push(``);
  console.log(`\nReview complete for ${completedBranches.length} branch(es).`);

  // -------------------------------------------------------------------------
  // Phase 4: Merge
  //
  // One agent merges all reviewed branches into the current branch,
  // resolving any conflicts and running tests to confirm everything still
  // works. The {{BRANCHES}} and {{ISSUES}} prompt arguments tell it which
  // branches to merge and which issues to close.
  // -------------------------------------------------------------------------
  await sandcastle.run({
    hooks,
    sandbox,
    name: "merger",
    maxIterations: 1,
    agent,
    promptFile: "./.sandcastle/merge-prompt.md",
    promptArgs: {
      // A markdown list of branch names, one per line.
      BRANCHES: completedBranches.map((b) => `- ${b}`).join("\n"),
      // A markdown list of issue IDs and titles, one per line.
      ISSUES: completedIssues.map((i) => `- ${i.id}: ${i.title}`).join("\n"),
    },
  });

  console.log("\nBranches merged.");
  iterLog.push(
    `### Merger`,
    ``,
    `Merged ${completedBranches.length} branch(es) into \`${launchBranch}\`.`,
    ``,
    `_Iteration duration: ${((Date.now() - iterStart) / 1000).toFixed(1)}s_`,
    ``,
  );
  iterationSummaries.push(...iterLog);
  writeSummary();
}

iterationSummaries.push(
  `---`,
  ``,
  `Run finished at ${new Date().toISOString()} (total ${((Date.now() - runStartTime) / 1000).toFixed(1)}s).`,
);
writeSummary();
console.log("\nAll done.");
console.log(`Run summary: ${summaryPath}`);
