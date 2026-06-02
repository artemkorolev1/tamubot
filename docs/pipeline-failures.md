# Pipeline failures — ratchet log

Append-only history of **real observed** v6b preprocessing failures and the fixes they
drove. Every algorithm change in the improvement loop must trace to an entry here (see the
`iterate-preprocessing` skill). The error taxonomy (`preprocessing_error_taxonomy.md`) is
the closed *vocabulary*; this file is the open *history*.

Newest first. One entry per distinct failure mode (not per syllabus instance).

## Template

```
### YYYY-MM-DD · <SHORT_TITLE> · <ERROR_TYPE>
- run: iter_<NN>_<sha>            stems: <example stems, 2-3>
- observed: <what the judge/human saw, with evidence — PDF page / chunk index>
- confirmed: <PDF-confirmed? recall@5 impact? anchor agreement?>
- owner: <code file from taxonomy>
- fix: <change made, or "open / not yet fixed">
- result: <post-fix paired-comparison delta; outside noise band? regressions?>
- status: open | fixed | wontfix (reason)
```

## Entries

### 2026-06-02 · Boilerplate reference under-populated (only CSCE+STAT materialized) · BP_MISSED
- run: _smoke    stems: 202611_CSCE_608_600_46648, 202611_CSCE_611_600_50668, 202611_CSCE_625_600_19180_HP
- observed: University-policy boilerplate is **not** flagged `is_boilerplate` (rate 0%);
  it either stays fully visible to RAG or surfaces as within-syllabus `is_duplicate`.
- confirmed: **two independent Claude Code sub-agent judges** flagged `BP_MISSED` (major)
  on CSCE_608 and CSCE_611, each citing the verbatim TAMU policy block (Academic Integrity,
  Title IX, ADA, FERPA, …) left in the processed view. Structural root cause:
  `meta_boilerplate_reference` gates on `distinct_depts >= 3`, but only CSCE+STAT are
  materialized, so no cluster qualifies. Not a tagging-logic bug.
- owner: data coverage (materialize ≥3 departments) before trusting `boilerplate_rate`;
  secondary: `util/boilerplate_clustering.py` gate constants.
- fix: open — materialize more departments, then rebuild `meta_boilerplate_reference`.
- result: —
- status: open
