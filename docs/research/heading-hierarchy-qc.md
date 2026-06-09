# Strengthening Quality-Control Gates for Heading-Hierarchy Reconstruction (v6b bronze stage)

> Research proposal. **No pipeline code is modified by this document.** It proposes a
> normalization algorithm + a gate set to replace the current ERROR-aborts-document
> behavior at `v6b_bronze_blocks`. Implementation is a follow-up, human-gated.

Date: 2026-06-07 · Scope: `v6b` bronze/converter stage (Docling → blocks.json + headers.json)

---

## 1. Problem restatement + root-cause confirmation

### 1.1 What breaks today

Three otherwise-clean syllabi (two ISEN, one CSCE) were thrown away in a recent run because a
single heading skipped a level (e.g. H1→H3). The mechanism, confirmed from source:

**The gate is a blocking ERROR.**
`src/tamubot/ingestion/pipeline_v6b/checks/bronze_blocks_checks.py:72-84`
(`v6b_bronze_blocks_header_hierarchy_valid`) is declared `blocking=True` with
`severity=AssetCheckSeverity.ERROR`. A blocking ERROR check failure aborts the partition, so
`v6b_silver_*` (chunk, tag, embed, upsert) never run for that stem — the whole syllabus is
dropped, not just its breadcrumb metadata.

**The rule it enforces is all-or-nothing.**
`src/tamubot/ingestion/validation/header_hierarchy.py:10-22` (`check_header_hierarchy_valid`)
returns `passed = (skip_count == 0)`. A *single* `level > prev_level + 1` transition anywhere
in the document fails the entire check. There is no rate, no severity gradient, no per-heading
attribution beyond a truncated `skip[:10]` list in metadata.

```python
# header_hierarchy.py:14-18 — one skip fails the doc
for h in headers:
    level = int(h.get("level", 0))
    if prev_level and level > prev_level + 1:
        skips.append(...)
    prev_level = level
```

### 1.2 Why the recovery pass does not prevent it

There *is* a recovery pass — `_recover_heading_levels` in
`src/tamubot/ingestion/converters/docling_block_adapter.py:292-326`. It does **not** enforce a
no-skip invariant:

- It two-pointer-aligns each heading block to `headers.json` (the font/TOC/numbering reference).
- **Matched** headers copy the reference level *verbatim* (`b["level"] = matched`, line 323) —
  so if the reference says H1 then H3, the blocks inherit H1 then H3. The skip survives.
- **Unmatched** headers are nested at `min(6, last_level + 1)` (line 326) — safe, never skips.

So the recovery handles the *unmatched* case correctly but trusts the reference for the
*matched* case, and the reference itself can skip.

### 1.3 Why the reference (`headers.json`) can skip

The reference is produced by `convert(..., apply_hierarchy=True)` in
`src/tamubot/ingestion/converters/docling_converter.py:121-242`, which runs the vendored
`ResultPostprocessor` (`_vendor/hierarchical/postprocessor.py`) then a
`_apply_hierarchy_safety_net` (`docling_converter.py:34-58`). Two places inject skips:

1. **The numbering inferrer trusts the document's own numbering depth.**
   `_vendor/hierarchical/hierarchy_builder.py:_infer_from_numbering` builds the tree from
   `1`, `1.1`, `1.1.1` patterns (`parsers.py:infer_header_level_numerical`). If a syllabus
   jumps from a `1.` section straight to a `1.1.1` item (author skipped `1.1`), the tree depth
   — and thus the level — jumps with it. `flatten_hierarchy_tree`
   (`postprocessor.py:39-45`) assigns `level = tree depth`, faithfully reproducing the skip.

2. **The font-clustering inferrer maps each DBSCAN cluster to a distinct level by font size.**
   `_cluster_headings_dbscan` (`hierarchy_builder.py:259-308`) sorts clusters by mean font
   size and assigns `cluster_to_level = {largest: 1, next: 2, ...}`. But
   `_infer_by_clustering` (`:205-257`) walks headings and re-parents by *relative* font size,
   and the safety net (`docling_converter.py:53-56`) independently **demotes** over-promoted
   H1/H2 lines to H3 (inline-label heuristic). A demote-to-H3 with no intervening H2 manufactures
   an H1→H3 skip directly in the final markdown that `headers.json` is derived from
   (`docling_converter.py:215-221`).

**Confirmed root cause:** the skip is a *symptom of three independent level-assignment paths*
(numbering depth, font cluster rank, safety-net demotion) that each produce locally-correct
relative ordering but no global no-skip guarantee — and the only consumer enforcing no-skip is a
hard ERROR gate with no repair step in front of it.

### 1.4 The conceptual trap (already established, restated for the proposal)

Detecting a skip tells you the levels are **inconsistent**, not **which** heading is wrong or
**how** to fix it. An H1→H3 skip can mean:

- **(a)** the H3 was over-deepened (common — safety-net demotion, numbering noise), or
- **(b)** a real H2 was genuinely missed upstream (Docling never emitted it).

A deterministic "clamp to `prev+1`" pass *always* yields a skip-free hierarchy and *never*
drops content, but it bets on **(a)**. If the truth is **(b)**, the heading lands one level too
shallow — and (Section 3) that costs only a slightly-wrong `header_path` breadcrumb, never a lost
section boundary or lost content.

---

## 2. Findings per research question

### RQ1 — Document hierarchy reconstruction best practices

Mature systems converge on the **same signal cascade**, and it is exactly the one this repo
vendors (`docling-hierarchical-pdf`):

| Priority | Signal | Reliability for syllabi | Source in repo |
|---|---|---|---|
| 1 | **PDF bookmarks / TOC** | Highest *when present* — author-declared truth. Rare in syllabi (most are flat 1-page exports). | `HierarchyBuilderMetadata.toc`, `postprocessor.py:168` |
| 2 | **Section numbering** (`1`, `1.1`, `A.`, `IV.`) | Strong when used, but syllabi number inconsistently (some sections numbered, most not). `_infer_from_numbering` bails if <30% numbered (`hierarchy_builder.py:100`). | `parsers.py`, `hierarchy_builder.py:80` |
| 3 | **Font size / weight clustering** | The workhorse for syllabi — most headings differ only by size/bold. DBSCAN over font size, clusters ranked by mean size. | `_cluster_headings_dbscan`, `hierarchy_builder.py:259` |
| 4 | **Indentation / position** | Weak signal; used only for false-positive cleanup (`cleanup_non_headings`, `hierarchy_builder.py:324` — drops repeated same-Y "headings" = running headers/footers). | same |

The package's own README states the cascade plainly: TOC via PyMuPDF bookmarks → numbering
(Arabic/roman/letter) → "infer the headings by font size and style (bold / italic)". It also
states the load-bearing limitation: *"if docling does not identify a header then there is no way
to get it back with this postprocessing."* That is precisely case **(b)** — a missed H2 cannot
be invented, only worked around. (Sources cited at end.)

**External corroboration.** Independent 2025 benchmarks (Procycons; HiPS textbook-segmentation
paper, arXiv 2509.00909) rank Docling as the most robust open framework for ToC reconstruction
vs Marker/GROBID/Unstructured, but all of them reconstruct hierarchy with the same TOC →
numbering → font/style ladder. None of them *guarantee* a no-skip outline; they recover a tree
and emit its depth. HiPS in particular frames hierarchy as **tree segmentation** (assign each
block a parent), which is the key reframing for RQ2.

**Syllabus-specific takeaway:** for *this* corpus, font-size clustering (signal 3) dominates,
numbering is partial, TOC is almost never present. So our repair must be robust to the
font-cluster path specifically — which is exactly the path that manufactures skips via
demotion.

### RQ2 — Repair vs flag: the right normalization algorithm

**The accessibility world flags; it never auto-repairs.** axe-core's `heading-order` rule and
WCAG tooling *detect* `heading levels should only increase by one` and hand it to a human
(Deque University; equalizedigital fix guide). Crucially, multiple WCAG sources note that a
literal skip is **not itself a WCAG 1.3.1 failure** — 1.3.1 is about whether the *programmatic
structure matches the visual/semantic structure*, and "skipping heading levels does not
represent a WCAG failure" per se (TPGi). **Implication for us:** the no-skip rule is a *useful
proxy for inconsistency*, not a correctness law. It is the right thing to *normalize and warn
on*, the wrong thing to *hard-fail on*.

**The smartest deterministic repair is tree-depth re-leveling, not naive clamp.**
There are three candidate normalizers, in increasing quality:

1. **Naive monotonic clamp.** `level = min(level, prev_emitted_level + 1)`. Skip-free,
   O(n), never drops content. Weakness: a single spurious deep heading "pins" the running
   level shallow and can *flatten legitimate depth that follows* (it only looks backward at the
   previous *emitted* level).

2. **Stack-based re-leveling (recommended).** Treat the level sequence as a *relative* ordering
   and re-emit **tree depth**. Maintain a monotonic stack of "open" levels; for each heading,
   pop levels `>=` its raw level, then emit `depth = len(stack) + 1`. This preserves *relative*
   nesting (a raw H3 under a raw H1 stays deeper than the H1) while compressing gaps so the
   emitted sequence is always skip-free — identical in spirit to the HTML5 document-outline
   algorithm and to BFS/DFS depth assignment over the implied section tree. It fixes case
   **(a)** perfectly and degrades gracefully on **(b)** (the orphaned-deep heading just becomes
   the next valid depth).

3. **Signal-aware re-leveling.** Same stack walk, but when a skip is detected, *consult the
   font-size / numbering signals already computed upstream* to decide (a) vs (b): if the skipped
   heading's font size is between its parent's and the next sibling's (suggesting a real missing
   middle tier), **insert a synthetic level** rather than clamp; otherwise clamp. This is the
   only normalizer that can recover case **(b)** depth — but it needs font metadata to survive
   into the blocks (today it does not; `headers.json` carries only `text`/`level`/`page`).

**Recommendation: ship #2 now, design #3 as a later enhancement.** #2 is pure-function,
testable, needs no new upstream data, and resolves 100% of *dropped documents* immediately. #3
is a quality refinement gated on plumbing font size into the reference (Section 4, Gate G5).

Pseudocode for #2 (stack-based re-leveling):

```text
def normalize_levels(headings):           # headings: [{text, level, ...}] in doc order
    out_levels = []
    stack = []                            # holds the *raw* levels currently "open"
    for h in headings:
        raw = max(1, int(h.level))
        # close any open sections at the same or deeper raw level
        while stack and stack[-1] >= raw:
            stack.pop()
        depth = len(stack) + 1            # 1-based, skip-free BY CONSTRUCTION
        stack.append(raw)
        out_levels.append(min(6, depth))  # cap at H6
    return out_levels
# Guarantees: out[i] - out[i-1] <= 1 for all i (no forward skip);
# relative ordering preserved; len(headings) unchanged (no drops).
```

Worked example (the failing case): raw `[1, 3, 3, 2, 4]` → normalized `[1, 2, 2, 2, 3]`. The
H1→H3 skip is gone; the two raw-H3 siblings stay siblings; the raw-H2 closes back up; the raw-H4
becomes a valid child. No content moved, only `#` counts changed.

### RQ3 — What mis-nesting actually costs downstream

Traced through `chunker_v4.py`:

- **Section tree.** `_parse_sections` (`chunker_v4.py:110-136`) builds the tree purely from the
  `#`-count *relative* ordering: `while stack[-1].level >= level: pop`. It is **already
  skip-tolerant** — an H1→H3 simply nests the H3 under the H1 (the intervening H2 absence is a
  no-op). So a skip never breaks chunk *boundaries* and never loses *content*.
- **`header_path`.** `_header_path` (`:142-143`) joins ancestor *header texts* with `>`. It uses
  the *text* of ancestors, not their numeric levels. A wrong level only changes **which
  ancestors appear in the breadcrumb** (e.g. an over-deepened heading inherits an extra
  ancestor, or a missed-H2 heading is missing one). The breadcrumb is embedded in chunk content
  for retrieval context but is *not* a hard boundary or filter key.
- **Oversized handling.** `chunk_semantic` (`:391-552`) recurses into children when a section is
  oversized (`tok > flag_threshold and node.children`). Re-leveling that turns a flat run into a
  proper parent/child tree *improves* this (more recursion points); flattening could merge a bit
  more under one parent but the `_split_long_text` leaf fallback (`:511-514`) still caps size.

**Quantified risk:** an imperfect repair costs *at most* a slightly-off breadcrumb on the
affected heading's subtree — a soft retrieval-context degradation, never a lost section, lost
table, or wrong chunk boundary. Compare to the **certain** cost today: the entire syllabus is
dropped (3 docs in one run). The asymmetry is stark — repair-and-warn strictly dominates
drop-the-doc. This also maps cleanly to the existing taxonomy: the *symptom* of a bad repair is
`FID_HEADER_BROKEN` (major, owner `assets/bronze_blocks.py`) and possibly `CHUNK_ORPHAN_HEADER`
(major), both of which the Inspect judge already scores per-doc — so we keep observability of
repair quality through the existing judge rather than the bronze ERROR gate.

### RQ4 — Quality-control gate design (the core deliverable)

See Section 3 for the full ranked proposal with files/lines/thresholds.

### RQ5 — Severity philosophy

**What should *ever* hard-fail a document at bronze?** Only conditions that mean *the extracted
text is unusable*, where shipping it would silently poison the index:

- `blocks.json` missing / empty (`v6b_bronze_blocks_nonempty`, already ERROR — keep).
- Replacement-char garbage above threshold (`..._no_replacement_chars`, already ERROR — keep:
  U+FFFD means decode failure, content *is* corrupt).

**What should never hard-fail:** structural-cosmetics that a deterministic pass can repair
without dropping content. Heading-level skips are the textbook example. The current
ERROR-aborts-doc on hierarchy is a **category error**: it treats a repairable cosmetic
inconsistency with the same severity as a corrupt parse. The fix is to **move repair upstream of
the check** and **downgrade the residual check to observability (WARN)**. This mirrors the
philosophy already encoded elsewhere in the bronze checks — `has_text`, `block_count_vs_baseline`,
and `source_integrity` are all `blocking=False` WARNs precisely because they are signals, not
correctness gates.

---

## 3. Prioritized, concrete proposal

### 3.1 Step 1 — Insert a normalization pass that *guarantees* a valid hierarchy

**Owning file:** `src/tamubot/ingestion/converters/docling_block_adapter.py`
(new pure function `normalize_heading_levels`, ~line 327, called at the end of
`_recover_heading_levels` and unconditionally in `docling_to_blocks` after recovery,
`docling_block_adapter.py:476-477`).

**What it does:** runs the RQ2 #2 stack-based re-leveling over the heading blocks *after*
`_recover_heading_levels`, mutating `b["level"]` in place. Because the chunker markdown and the
phase-2 `header_path` are both derived from these block levels, fixing it here fixes everything
downstream with no re-Docling.

**Guarantee:** output is skip-free by construction, content count unchanged. This *alone*
eliminates the dropped-document failure even before any gate changes.

**Also normalize `headers.json`.** The sidecar at
`docling_converter.py:211-232` is consumed by `_recover_heading_levels` and by chunk-page
anchoring. Apply the same re-leveling to the sidecar list (or have `_recover_heading_levels`
re-level after alignment). Keep the *pre-normalization* raw levels available for Gate G2's
metric (so we can still count "how many skips did we repair").

### 3.2 Step 2 — Replace the ERROR gate with an observability gate

**Owning file:** `src/tamubot/ingestion/pipeline_v6b/checks/bronze_blocks_checks.py:72-84`
and the validator `src/tamubot/ingestion/validation/header_hierarchy.py`.

After Step 1, `check_header_hierarchy_valid` on the *normalized* levels can never fail — so the
blocking ERROR is dead weight. Repurpose it as a **post-condition assertion** (L1) and add a
**repair-observability** check (L2) that preserves the signal the ERROR gate used to give.

### 3.3 The gate set (ranked by priority)

Levels follow the repo's L1/L2/L3 convention (`ingestion/CLAUDE.md:34`): **L1** = per-partition
structural invariant; **L2** = per-partition metric with run-over-run baseline delta; **L3** =
corpus/quality (opt-in, expensive).

| # | Gate | What it measures | Metric / threshold | Severity | Owning file:line | Surfaces as |
|---|---|---|---|---|---|---|
| **G1** | `header_hierarchy_valid` *(repurposed)* | Post-normalization invariant: `skip_count == 0` on the *normalized* levels | `skip_count == 0` (must always hold) | **L1, WARN, blocking=False** (was ERROR/blocking) | `bronze_blocks_checks.py:72`; validator `header_hierarchy.py:10` | bronze "Checks" tab |
| **G2** | `header_levels_normalized` *(new)* | How much repair happened — keeps the signal the ERROR gate gave us | `repaired_skip_count` (raw skips fixed); WARN if `> 0`, ERROR never | **L2, WARN** | new fn in `header_hierarchy.py`; check in `bronze_blocks_checks.py` | bronze tab + metadata `repaired_skips[:10]` |
| **G3** | `suspicious_heading_rate` *(new)* | Root cause: bold body-line promoted to heading (false positive) | fraction of headings that are inline-label-shaped (`_INLINE_LABEL_RE`) or > ~12 words or end with sentence punctuation; WARN if `> 0.15` | **L2, WARN** | new validator `header_hierarchy.py`; uses block text | bronze tab |
| **G4** | `min_headers` *(existing helper, wire it)* | Root cause: real headings demoted to body (the doc came out nearly flat) | `header_count >= 2` AND distinct-levels `>= 2` when `page_count > 1` | **L1, WARN** | `check_min_headers` (`header_hierarchy.py:25`) — currently unused; wire into `bronze_blocks_checks.py` | bronze tab |
| **G5** | `repair_confidence` *(new, future — needs font plumbing)* | Did the repair likely hit case (b) (missed real H2)? | count of skips where the demoted heading's font size sits *between* parent and child sizes → "structural-loss suspected" | **L2/L3, WARN** | `header_hierarchy.py` + plumb `font_size` into `headers.json` (`docling_converter.py:221`) | bronze tab + judge cross-check |
| **G6** | `heading_repair_vs_baseline` *(new)* | Drift watchdog: a code change that suddenly repairs 10x more skips is a regression signal | `repaired_skip_count` via `compute_baseline_delta(..., max_drift_pct=0.20)` | **L2, WARN** | `bronze_blocks_checks.py` using `baseline_diff.py` (same pattern as `block_count_vs_baseline:87`) | bronze tab (delta %) |

**Ranking rationale:**

1. **G1 + Step-1 normalization** — *must ship together*. This is the entire ask: stop dropping
   documents. G1 downgraded to WARN is only safe *because* Step 1 guarantees it passes; ship the
   normalizer first, flip the severity second, in one PR.
2. **G2** — highest-value *new* observability. Without it, lowering G1 to WARN throws away the
   only signal we had that a doc's hierarchy was off. G2 is that signal, now non-fatal.
3. **G4** — cheapest root-cause guard (helper already exists, just unused). Catches the
   "everything demoted to body" failure that the no-skip rule *cannot* see (a fully-flat doc has
   zero skips).
4. **G3** — catches the opposite root cause (body promoted to heading). Reuses the
   `_INLINE_LABEL_RE` heuristic already in `docling_converter.py:31`.
5. **G6** — regression watchdog, reuses the existing baseline-delta machinery.
6. **G5** — best-quality but most expensive (needs font metadata plumbed through bronze); defer
   to a follow-up once Steps 1–2 land.

### 3.4 How this replaces the current ERROR gate

- **Before:** one heading skip → `check_header_hierarchy_valid` fails → blocking ERROR → silver
  never runs → syllabus dropped. Signal: binary, fatal, no repair.
- **After:** Step-1 normalizer guarantees no skips → G1 (now WARN/non-blocking) always passes →
  silver runs → syllabus indexed. The *information* "this doc had N raw skips / M suspicious
  headings" is preserved as WARN metadata on G2/G3/G4/G6 and remains visible in the Dagster
  Checks tab and to the Inspect judge (`FID_HEADER_BROKEN`). **No document is ever dropped for a
  cosmetic hierarchy glitch again.**

---

## 4. Trade-offs of the chosen repair vs alternatives

| Approach | Drops docs? | Fixes case (a) | Recovers case (b) | New upstream data? | Verdict |
|---|---|---|---|---|---|
| **Status quo** (ERROR gate, no repair) | **Yes (the bug)** | — | — | no | Reject |
| Naive monotonic clamp | No | Yes | No | no | Acceptable, but can flatten legit depth |
| **Stack-based re-leveling (#2, chosen)** | No | Yes | No (degrades gracefully) | no | **Recommended now** |
| Signal-aware re-leveling (#3 / G5) | No | Yes | **Partial** (insert synthetic level) | yes (font size in `headers.json`) | Recommended later |
| LLM/judge re-leveling | No | Yes | Maybe | LLM call per doc | Reject for bronze (cost, non-determinism, CLAUDE.md ≤10-call rule) |

**Chosen-repair trade-offs (stack-based #2):**

- **Pro:** pure, deterministic, O(n), unit-testable in isolation (host-Python friendly — no
  Docling import needed, satisfies the "isolated module imports work" constraint in MEMORY). No
  new upstream signals. Provably skip-free and content-preserving. Eliminates the drop bug
  outright.
- **Con (accepted):** on a genuine missing-H2 (case **b**), the orphaned-deep heading is
  re-leveled one tier too shallow → a slightly-wrong `header_path` breadcrumb on that subtree.
  Section 3 quantifies this as a soft retrieval-context degradation, never a lost boundary or
  lost content — and G2/G5 keep it *observable*, so we can decide later whether case (b) is
  frequent enough to justify shipping #3.
- **Con (accepted):** re-leveling changes `#` counts in the bronze markdown, which will move the
  `block_count_vs_baseline` / chunk baselines on the first run after deploy. Expected one-time
  drift; the WARN baselines will re-settle (and G6 explicitly watches for it).

**Why not just lower the existing gate to WARN and skip the normalizer?** Because the chunker and
`header_path` would still consume skip-laden levels. You'd stop dropping docs but ship subtly
wrong breadcrumbs *with no repair and no observability*. The normalizer is the load-bearing
part; the gate flip is the cleanup.

---

## Sources (external claims)

- docling-hierarchical-pdf (the vendored package) — signal cascade + "if docling does not
  identify a header there is no way to get it back": <https://github.com/krrome/docling-hierarchical-pdf>
- WCAG 1.3.1 / skipped headings are not per-se failures (semantic-match is the real rule),
  TPGi: <https://www.tpgi.com/heading-off-confusion-when-do-headings-fail-wcag/>
- "Heading levels should only increase by one" (the rule we proxy), Deque/axe:
  <https://dequeuniversity.com/rules/axe/4.4/heading-order> ·
  <https://equalizedigital.com/accessibility-checker/incorrect-heading-order/>
- axe-core heading-order *flags, does not repair*; misses some structural cases:
  <https://github.com/dequelabs/axe-core/issues/220>
- Docling vs Marker/GROBID/Unstructured ToC reconstruction benchmark (2025), Procycons:
  <https://procycons.com/en/blogs/pdf-data-extraction-benchmark/>
- HiPS: Hierarchical PDF Segmentation (hierarchy = tree segmentation), arXiv 2509.00909:
  <https://arxiv.org/html/2509.00909v1>
- HTML5 document-outline algorithm / level = nesting depth (rationale for tree-depth re-leveling),
  Heydon: <https://medium.com/@Heydon/managing-heading-levels-in-design-systems-18be9a746fa3>
