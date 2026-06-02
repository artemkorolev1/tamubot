// Parallel Planner — human-gated, per-stage orchestration.
//
// A single invocation runs ONE round:
//   Phase 1 (Plan):    An opus agent reads `ready-for-agent` issues, builds a
//                      dependency graph, and emits a <plan> of unblocked issues.
//   Phase 2 (Execute): N sonnet agents in parallel (Promise.allSettled), each
//                      working one issue on its own `sandcastle/issue-{id}` branch.
//   Phase 3 (Review):  One sonnet agent per completed branch posts a verdict
//                      comment on the issue.
//   STOP — nothing is merged. A merge plan is printed; YOU decide what merges.
//   Phase 4 (Merge):   Run separately, after you approve, via the `merge` mode.
//
// Models and skills are per-role (see the configuration block below): Opus for
// planning/merging, Sonnet for the parallel implementers/reviewers; each role
// gets its own curated skill mount so a coding agent isn't handed unrelated
// domain skills.
//
// Two human gates: you approve the PLAN, then you approve the MERGE.
//
// Usage:
//   npm run sandcastle                         # GATE 1: plan only — show the plan, then stop
//   npm run sandcastle -- execute              # run the approved plan (execute → review), stop at merge gate
//   npm run sandcastle -- merge <branch...>    # GATE 2: human-approved merge of the listed branches
//   npm run sandcastle -- merge                # dry-run: list mergeable branches, merge nothing
//   npm run sandcastle -- auto                 # no plan gate: plan + execute + review in one shot
//   npm run sandcastle -- implement <id> [title]   # re-run a single implementer
//   npm run sandcastle -- review <id>              # re-run a single reviewer
//
// (The bare script is `node --env-file=.sandcastle/.env --import tsx .sandcastle/main.mts`.)

import * as sandcastle from "@ai-hero/sandcastle";
import { docker } from "@ai-hero/sandcastle/sandboxes/docker";
import { z } from "zod";
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
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

// Pre-flight: refuse to run on a dirty working tree. Per-issue worktrees branch
// off HEAD; uncommitted host changes would either be invisible to the agents or
// get captured mid-run. The merge mode also needs a clean tree to merge into.
// Override with SANDCASTLE_ALLOW_DIRTY=1 when you knowingly want to proceed.
const dirty = execSync("git status --porcelain", { encoding: "utf8" }).trim();
if (dirty && process.env.SANDCASTLE_ALLOW_DIRTY !== "1") {
  throw new Error(
    `Working tree is dirty — commit, stash, or set SANDCASTLE_ALLOW_DIRTY=1.\n\n${dirty}\n`,
  );
}
const launchBranch = execSync("git branch --show-current", { encoding: "utf8" }).trim();

// ---------------------------------------------------------------------------
// Configuration — models
// ---------------------------------------------------------------------------
// Planning and merging are the high-stakes reasoning phases (dependency graphs;
// conflict resolution + keeping the integrated tree green), so they run on Opus.
// Implementers and reviewers are well-scoped and far more numerous, so they run
// on Sonnet. Override any of these with the SANDCASTLE_*_MODEL env vars.
const PLANNER_MODEL = process.env.SANDCASTLE_PLANNER_MODEL ?? "claude-opus-4-8";
const IMPLEMENTER_MODEL = process.env.SANDCASTLE_IMPLEMENTER_MODEL ?? "claude-sonnet-4-6";
const REVIEWER_MODEL = process.env.SANDCASTLE_REVIEWER_MODEL ?? "claude-sonnet-4-6";
const MERGER_MODEL = process.env.SANDCASTLE_MERGER_MODEL ?? "claude-opus-4-8";

const plannerAgent = sandcastle.claudeCode(PLANNER_MODEL);
const implementerAgent = sandcastle.claudeCode(IMPLEMENTER_MODEL);
const reviewerAgent = sandcastle.claudeCode(REVIEWER_MODEL);
const mergerAgent = sandcastle.claudeCode(MERGER_MODEL);

// ---------------------------------------------------------------------------
// Configuration — per-stage skills  ◀── EDIT HERE to change what each role can do
// ---------------------------------------------------------------------------
// Each list names directories under the HOST's ~/.claude/skills. The matching
// skill is bind-mounted (read-write) into that role's sandbox at
// /home/agent/.claude/skills/<name>, so only the listed skills are visible to
// that stage. Edits an agent makes to a skill write back to the host and persist
// into the next run. A name that doesn't resolve to a host dir is skipped with a
// warning (so you can leave aspirational entries in the list).
const PLANNER_SKILLS = ["task-budget", "to-prd", "grill-with-docs"];
const IMPLEMENTER_SKILLS = [
  "tdd",
  "diagnose",
  "probe-rag",
  "run-eval",
  "langfuse",
  "server-ops",
  "task-budget",
];
const REVIEWER_SKILLS = ["improve-codebase-architecture", "diagnose"];
const MERGER_SKILLS = ["task-budget"];

// Implementer iteration cap. Most well-scoped issues converge in <10 turns; a
// misbehaving agent that loops forever costs real budget — keep this tight.
const IMPLEMENTER_MAX_ITERATIONS = Number(process.env.SANDCASTLE_IMPLEMENTER_MAX_ITERATIONS ?? 30);

// ---------------------------------------------------------------------------
// Per-role sandboxes (curated skill mounts)
// ---------------------------------------------------------------------------
type Mount = { hostPath: string; sandboxPath: string; readonly?: boolean };

const HOST_SKILLS_DIR = join(homedir(), ".claude", "skills");

function skillMounts(skillNames: readonly string[]): Mount[] {
  const mounts: Mount[] = [];
  for (const name of skillNames) {
    const hostPath = join(HOST_SKILLS_DIR, name);
    if (existsSync(hostPath)) {
      mounts.push({ hostPath, sandboxPath: `~/.claude/skills/${name}` });
    } else {
      console.warn(`  ⚠ skill "${name}" not found at ${hostPath} — skipping mount`);
    }
  }
  return mounts;
}

function sandboxWithSkills(skillNames: readonly string[]) {
  return docker({
    imageName: "tamubot-sandcastle:local",
    env: { CLAUDE_CODE_OAUTH_TOKEN: oauthToken!, GH_TOKEN: ghToken! },
    mounts: skillMounts(skillNames),
  });
}

const plannerSandbox = sandboxWithSkills(PLANNER_SKILLS);
const implementerSandbox = sandboxWithSkills(IMPLEMENTER_SKILLS);
const reviewerSandbox = sandboxWithSkills(REVIEWER_SKILLS);
const mergerSandbox = sandboxWithSkills(MERGER_SKILLS);

const hooks = {
  sandbox: { onSandboxReady: [{ command: "echo sandbox ready" }] },
};

// The planner emits its plan as JSON inside <plan> tags; Output.object extracts
// and validates it against this schema.
const planSchema = z.object({
  issues: z.array(
    z.object({ id: z.string(), title: z.string(), branch: z.string() }),
  ),
});

type Issue = { id: string; title: string; branch: string };

// ---------------------------------------------------------------------------
// Logging / run artifacts
// ---------------------------------------------------------------------------
const runStartedAt = new Date().toISOString().replace(/[:.]/g, "-");
const runsDir = join(".sandcastle", "logs", "runs");
const runLogDir = join(runsDir, runStartedAt);
mkdirSync(runLogDir, { recursive: true });
const summaryPath = join(runsDir, `${runStartedAt}.md`);
const summary: string[] = [];
const writeSummary = () => writeFileSync(summaryPath, summary.join("\n") + "\n");

// Predictable, per-run log path so each agent's full transcript is easy to find
// and correlate. Returned to run() as `logging`; the same path is recorded in
// the summary for observability.
function logFor(role: string, key?: string) {
  const name = key ? `${role}-${key}` : role;
  return { type: "file" as const, path: join(runLogDir, `${name}.log`) };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const branchFor = (id: string) => `sandcastle/issue-${id}`;
const idFromBranch = (b: string) => b.match(/issue-(\d+)/)?.[1] ?? b;

// Where `plan` saves the approved plan for a later `execute` to pick up.
const PLAN_FILE = join(".sandcastle", "logs", "last-plan.json");

function loadPlan(): { launchBranch: string; issues: Issue[] } {
  if (!existsSync(PLAN_FILE)) {
    throw new Error(`No saved plan at ${PLAN_FILE}. Run \`npm run sandcastle -- plan\` first.`);
  }
  return JSON.parse(readFileSync(PLAN_FILE, "utf8"));
}

function issueTitle(id: string): string {
  try {
    return execSync(`gh issue view ${id} --json title --jq .title`, { encoding: "utf8" }).trim();
  } catch {
    return `issue ${id}`;
  }
}

function parseVerdict(text: string): string {
  if (/\*\*Changes requested\*\*/i.test(text)) return "Changes requested";
  if (/\*\*LGTM with notes\*\*/i.test(text)) return "LGTM with notes";
  if (/\*\*LGTM\*\*/i.test(text)) return "LGTM";
  return "unknown";
}

// The reviewer's verdict lives in the issue comment it posts (the `gh` tool-call
// body), which sandcastle routes to the log file but NOT to RunResult.stdout —
// so we read it back from the posted comment, the source of truth. Pick the last
// comment authored by the reviewer agent.
function fetchVerdict(id: string): string {
  try {
    const body = execSync(
      `gh issue view ${id} --json comments ` +
        `--jq '[.comments[] | select(.body | contains("Sandcastle reviewer agent"))] | last | .body // ""'`,
      { encoding: "utf8" },
    );
    return parseVerdict(body);
  } catch {
    return "unknown";
  }
}

// ---------------------------------------------------------------------------
// Stage runners (shared by the full round and the single-issue re-run modes)
// ---------------------------------------------------------------------------
function runImplementer(issue: Issue) {
  return sandcastle.run({
    hooks,
    sandbox: implementerSandbox,
    branchStrategy: { type: "branch", branch: issue.branch },
    name: "implementer",
    maxIterations: IMPLEMENTER_MAX_ITERATIONS,
    agent: implementerAgent,
    promptFile: "./.sandcastle/implement-prompt.md",
    logging: logFor("implementer", issue.id),
    promptArgs: { TASK_ID: issue.id, ISSUE_TITLE: issue.title, BRANCH: issue.branch },
  });
}

function runReviewer(issue: Issue) {
  return sandcastle.run({
    hooks,
    sandbox: reviewerSandbox,
    branchStrategy: { type: "branch", branch: issue.branch },
    name: "reviewer",
    maxIterations: 20,
    agent: reviewerAgent,
    promptFile: "./.sandcastle/review-prompt.md",
    logging: logFor("reviewer", issue.id),
    promptArgs: { TASK_ID: issue.id, ISSUE_TITLE: issue.title, BRANCH: issue.branch },
  });
}

// ---------------------------------------------------------------------------
// Mode: round — plan → execute → review, then STOP with a merge plan.
// ---------------------------------------------------------------------------
// --- Gate 1: Plan only. Run the planner, save the plan, print it, STOP. ---
async function runPlan(): Promise<Issue[]> {
  console.log(`Launch branch: ${launchBranch}`);
  console.log(
    `Models — planner: ${PLANNER_MODEL}, implementer: ${IMPLEMENTER_MODEL}, ` +
      `reviewer: ${REVIEWER_MODEL}, merger: ${MERGER_MODEL}`,
  );
  summary.push(
    `# Sandcastle plan — ${runStartedAt}`,
    ``,
    `- Launch branch: \`${launchBranch}\``,
    `- Models: planner \`${PLANNER_MODEL}\`, implementer \`${IMPLEMENTER_MODEL}\`, reviewer \`${REVIEWER_MODEL}\`, merger \`${MERGER_MODEL}\``,
    `- Implementer max iterations: ${IMPLEMENTER_MAX_ITERATIONS}`,
    `- Logs: \`${runLogDir}\``,
    ``,
  );
  writeSummary();

  console.log(`\n=== Phase 1: Plan ===\n`);
  const plan = await sandcastle.run({
    hooks,
    sandbox: plannerSandbox,
    name: "planner",
    maxIterations: 1,
    agent: plannerAgent,
    promptFile: "./.sandcastle/plan-prompt.md",
    logging: logFor("planner"),
    output: sandcastle.Output.object({ tag: "plan", schema: planSchema }),
  });
  const issues = plan.output.issues;

  summary.push(`## Plan`, ``, `Planner log: \`${plan.logFilePath ?? logFor("planner").path}\``, ``);
  if (issues.length === 0) {
    console.log("No unblocked issues to work on.");
    summary.push(`Planner found 0 unblocked issues — nothing to do.`, ``);
    writeSummary();
    return issues;
  }
  console.log(`${issues.length} issue(s) selected:`);
  summary.push(`Selected ${issues.length} issue(s):`, ``);
  for (const issue of issues) {
    console.log(`  #${issue.id}: ${issue.title} → ${issue.branch}`);
    summary.push(`- #${issue.id} "${issue.title}" → \`${issue.branch}\``);
  }
  summary.push(``);

  // Persist so a later `execute` invocation can pick this plan up.
  writeFileSync(PLAN_FILE, JSON.stringify({ savedAt: runStartedAt, launchBranch, issues }, null, 2));
  summary.push(`Plan saved to \`${PLAN_FILE}\`.`, ``);
  writeSummary();

  return issues;
}

// --- Phases 2+3: Execute the approved plan, then review. Stops at merge gate. ---
async function runExecuteReview(issues: Issue[]) {
  if (issues.length === 0) {
    console.log("Plan is empty — nothing to execute.");
    return;
  }
  if (summary.length === 0) {
    // Standalone `execute` invocation — seed a report header.
    summary.push(
      `# Sandcastle execute — ${runStartedAt}`,
      ``,
      `- Launch branch: \`${launchBranch}\``,
      `- Executing ${issues.length} planned issue(s): ${issues.map((i) => `#${i.id}`).join(", ")}`,
      `- Logs: \`${runLogDir}\``,
      ``,
    );
    writeSummary();
  }
  // --- Phase 2: Execute ---
  console.log(`\n=== Phase 2: Execute (${issues.length} in parallel) ===\n`);
  const settled = await Promise.allSettled(issues.map((issue) => runImplementer(issue)));

  summary.push(`## Implementers`, ``);
  const completedIssues: Issue[] = [];
  for (const [i, outcome] of settled.entries()) {
    const issue = issues[i]!;
    const logPath = logFor("implementer", issue.id).path;
    if (outcome.status === "rejected") {
      console.error(`  ✗ ${issue.id} (${issue.branch}) failed: ${outcome.reason}`);
      summary.push(`- #${issue.id} (${issue.branch}) — **failed**: \`${String(outcome.reason).slice(0, 200)}\` — log: \`${logPath}\``);
    } else {
      const n = outcome.value.commits.length;
      summary.push(`- #${issue.id} (${issue.branch}) — ${n} commit(s) — log: \`${logPath}\``);
      if (n > 0) completedIssues.push(issue);
    }
  }
  summary.push(``);
  writeSummary();

  if (completedIssues.length === 0) {
    console.log("No commits produced. Nothing to review or merge.");
    summary.push(`No commits produced — no review or merge plan.`, ``);
    writeSummary();
    return;
  }

  // --- Phase 3: Review ---
  console.log(`\n=== Phase 3: Review (${completedIssues.length}) ===\n`);
  const reviewSettled = await Promise.allSettled(completedIssues.map((issue) => runReviewer(issue)));

  const verdicts = new Map<string, string>();
  summary.push(`## Reviewers`, ``);
  for (const [i, outcome] of reviewSettled.entries()) {
    const issue = completedIssues[i]!;
    const logPath = logFor("reviewer", issue.id).path;
    if (outcome.status === "rejected") {
      console.error(`  ✗ review of ${issue.id} failed: ${outcome.reason}`);
      verdicts.set(issue.id, "review failed");
      summary.push(`- #${issue.id} — **review failed**: \`${String(outcome.reason).slice(0, 200)}\` — log: \`${logPath}\``);
    } else {
      const verdict = fetchVerdict(issue.id);
      verdicts.set(issue.id, verdict);
      summary.push(`- #${issue.id} — verdict: **${verdict}** — log: \`${logPath}\``);
    }
  }
  summary.push(``);

  // --- STOP: print the merge plan; do NOT merge ---
  const cmd = `npm run sandcastle -- merge ${completedIssues.map((i) => i.branch).join(" ")}`;
  console.log(`\n=== Review complete — MERGE IS GATED ON YOU ===\n`);
  console.log(`Branches ready for your decision:`);
  for (const issue of completedIssues) {
    console.log(`  ${issue.branch}  [${verdicts.get(issue.id)}]   inspect: git diff ${launchBranch}..${issue.branch}`);
  }
  console.log(`\nWhen you've approved, merge the ones you want:\n  ${cmd}\n`);

  summary.push(
    `## Merge plan (gated — not yet merged)`,
    ``,
    ...completedIssues.map(
      (i) => `- \`${i.branch}\` — verdict **${verdicts.get(i.id)}** — inspect: \`git diff ${launchBranch}..${i.branch}\``,
    ),
    ``,
    `Approve and merge with:`,
    ``,
    "```",
    cmd,
    "```",
    ``,
  );
  writeSummary();
  console.log(`Run summary: ${summaryPath}`);
}

// ---------------------------------------------------------------------------
// Mode: merge — human-approved merge of explicitly listed branches.
// ---------------------------------------------------------------------------
async function runMerge(branches: string[]) {
  if (branches.length === 0) {
    // Dry run: list candidate branches ahead of the launch branch; merge nothing.
    const raw = execSync('git for-each-ref --format="%(refname:short)" refs/heads/sandcastle', {
      encoding: "utf8",
    }).trim();
    const candidates = raw ? raw.split("\n") : [];
    console.log(`No branches given — nothing merged. Candidate branches:`);
    if (candidates.length === 0) {
      console.log(`  (none found under sandcastle/*)`);
      return;
    }
    for (const b of candidates) {
      let ahead = "?";
      try {
        ahead = execSync(`git rev-list --count ${launchBranch}..${b}`, { encoding: "utf8" }).trim();
      } catch {
        /* ignore */
      }
      console.log(`  ${b}  (+${ahead} commit(s) over ${launchBranch})`);
    }
    console.log(`\nMerge with:\n  npm run sandcastle -- merge ${candidates.join(" ")}`);
    return;
  }

  const issues: Issue[] = branches.map((b) => {
    const id = idFromBranch(b);
    return { id, title: issueTitle(id), branch: b };
  });

  console.log(`Merging ${branches.length} branch(es) into ${launchBranch}:`);
  for (const i of issues) console.log(`  ${i.branch} (#${i.id} ${i.title})`);

  await sandcastle.run({
    hooks,
    sandbox: mergerSandbox,
    name: "merger",
    maxIterations: 1,
    agent: mergerAgent,
    promptFile: "./.sandcastle/merge-prompt.md",
    logging: logFor("merger"),
    promptArgs: {
      BRANCHES: branches.map((b) => `- ${b}`).join("\n"),
      ISSUES: issues.map((i) => `- ${i.id}: ${i.title}`).join("\n"),
    },
  });

  console.log(`\nMerge complete. Log: ${logFor("merger").path}`);
}

// ---------------------------------------------------------------------------
// Modes: single-issue re-runs (surgical retries without the full pipeline).
// ---------------------------------------------------------------------------
async function reRunImplementer(id: string, title?: string) {
  const issue: Issue = { id, title: title ?? issueTitle(id), branch: branchFor(id) };
  console.log(`Re-running implementer for #${id} on ${issue.branch}`);
  const r = await runImplementer(issue);
  console.log(`Done. ${r.commits.length} commit(s). Log: ${logFor("implementer", id).path}`);
}

async function reRunReviewer(id: string, title?: string) {
  const issue: Issue = { id, title: title ?? issueTitle(id), branch: branchFor(id) };
  console.log(`Re-running reviewer for #${id} on ${issue.branch}`);
  await runReviewer(issue);
  console.log(`Done. Verdict: ${fetchVerdict(id)}. Log: ${logFor("reviewer", id).path}`);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
const [, , mode, ...rest] = process.argv;

switch (mode) {
  case undefined:
  case "plan": {
    // Gate 1: plan only. Show the plan and stop; nothing is executed.
    const issues = await runPlan();
    if (issues.length > 0) {
      console.log(`\n=== Plan ready — EXECUTION IS GATED ON YOU ===`);
      console.log(`Review the plan above, then run:\n  npm run sandcastle -- execute\n`);
    }
    break;
  }
  case "execute": {
    // Gate 2 entry: run the approved plan (execute + review), stop at merge gate.
    const { launchBranch: planned, issues } = loadPlan();
    if (planned !== launchBranch) {
      console.warn(
        `⚠ Saved plan was made on "${planned}" but you are on "${launchBranch}". ` +
          `Re-run \`plan\` if that's not intended.`,
      );
    }
    await runExecuteReview(issues);
    break;
  }
  case "auto": {
    // No plan gate: plan, then immediately execute + review (still stops at merge).
    const issues = await runPlan();
    await runExecuteReview(issues);
    break;
  }
  case "merge":
    await runMerge(rest);
    break;
  case "implement":
    if (!rest[0]) throw new Error("Usage: implement <issue-id> [title]");
    await reRunImplementer(rest[0], rest.slice(1).join(" ") || undefined);
    break;
  case "review":
    if (!rest[0]) throw new Error("Usage: review <issue-id> [title]");
    await reRunReviewer(rest[0], rest.slice(1).join(" ") || undefined);
    break;
  default:
    throw new Error(
      `Unknown mode "${mode}". Use: (no arg)|plan | execute | auto | merge | implement | review`,
    );
}

console.log("\nDone.");
