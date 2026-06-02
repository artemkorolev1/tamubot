"""
Download syllabi from tamu.simplesyllabus.com.

Uses Playwright to acquire session cookies (site is JS-gated behind CloudFront),
then fetches the search API and prints each syllabus page to PDF.

Output structure:
  tamu_data/raw/simple_syllabus/{SUBJECT}/{Term_Year}/
    {term_code}_{SUBJECT}_{COURSE}_{SECTION}_{CRN}.pdf
    metadata.json
"""

import json
import os
import random
import re
import time
from urllib.parse import quote

from playwright.sync_api import sync_playwright

# ── Config ───────────────────────────────────────────────────────────────────

DEPARTMENTS = {"CSCE", "ISEN", "STAT", "ECEN"}
GRADUATE_ONLY = True  # course number >= 600
TARGET_TERMS = {"Spring 2026", "Summer 2026", "Fall 2026"}
PAGE_SIZE = 50
# Pacing is env-tunable so a broad crawl can slow down to stay under CloudFront's WAF
# rate limits on the API path (which returns 403 HTML error pages when tripped).
DELAY = float(os.getenv("SIMPLE_SYLLABUS_DELAY", "1.0"))  # seconds between requests
MAX_RETRIES = int(os.getenv("SIMPLE_SYLLABUS_MAX_RETRIES", "5"))

# Optional cellular-data budget for running over a metered phone hotspot: stop once estimated
# network use reaches this many MB (0 = unlimited). The view page render costs more than the
# saved PDF, so we approximate network ≈ output_bytes × NET_FACTOR. Deliberately conservative
# (overestimates → stops early rather than overshooting the data cap). Resumable: re-running
# skips already-downloaded files, so a stop is safe.
MAX_MB = float(os.getenv("SIMPLE_SYLLABUS_MAX_MB", "0"))
NET_FACTOR = float(os.getenv("SIMPLE_SYLLABUS_NET_FACTOR", "2.5"))

# SIMPLE_SYLLABUS_DEPTS overrides the department set with a comma-separated list
# (e.g. "CSCE,ECEN,ACCT"). Used for the broad "all departments" golden-set crawl, where the
# list is derived from the Howdy Portal scrape: the Simple Syllabus API requires a non-empty
# `search=` value (an empty search returns an HTML error page, not JSON), so it must be queried
# one department at a time. The default set is otherwise left intact for normal runs.
_DEPTS_ENV = os.getenv("SIMPLE_SYLLABUS_DEPTS", "").strip()
if _DEPTS_ENV:
    DEPARTMENTS = {d.strip().upper() for d in _DEPTS_ENV.split(",") if d.strip()}

BASE = "https://tamu.simplesyllabus.com"
API = f"{BASE}/api2/doc-library-search"

# Term entity_ids — filter server-side, avoids scanning all 11k items
_TERM_IDS = {
    "Summer 2025": "9ca19ce6-e67d-485f-b6d4-3febcab32aa5",
    "Fall 2025": "ecd304d6-7795-4f49-a0ee-1d6137884ac7",
    "Spring 2026": "3a9c109e-8e72-4682-966e-b6c754c0596f",
    "Summer 2026": "1da0b525-6f1a-42f5-a614-51a5b9e4b3eb",
    "Fall 2026": "6dcfa515-e3c9-4cd0-8413-ff8c00517ae3",
}
_TERM_ID_QS = "&".join(f"term_ids[]={tid}" for t, tid in _TERM_IDS.items() if t in TARGET_TERMS)
# Use search= to pre-filter by department name server-side. The API only honors a
# single search= value, so scan once per department and union the results.
PARAMS = f"{_TERM_ID_QS}&search={{dept}}&page_size={{ps}}&page={{pg}}"

TERM_SEMESTER = {"spring": "11", "summer": "21", "fall": "41"}

# Some terms use raw codes (e.g. "202521") instead of "Summer 2025 - College Station"
_CODE_TO_TERM = {"202521": "Summer 2025"}

RAW_ROOT = os.path.join("tamu_data", "raw", "simple_syllabus")

# Title pattern: "CSCE 670 600 (46627)"
_TITLE_RE = re.compile(r"^(?P<subject>[A-Z]+)\s+(?P<course>\d+)\s+(?P<section>\S+)\s+\((?P<crn>\d+)\)$")

# ── Helpers ───────────────────────────────────────────────────────────────────


def term_to_code(term: str) -> str:
    """'Spring 2026' → '202611'"""
    parts = term.lower().split()
    return f"{parts[1]}{TERM_SEMESTER.get(parts[0], '00')}" if len(parts) == 2 else "unknown"


def term_to_folder(term: str) -> str:
    """'Spring 2026' → 'Spring_2026'"""
    return term.replace(" ", "_")


def term_to_slug_prefix(term_name: str) -> str:
    """'Spring 2026 - College Station' → 'Spring-2026-College-Station'"""
    return term_name.replace(" - ", "-").replace(" ", "-")


def build_slug(term_name: str, subject: str, course: str, section: str, crn: str) -> str:
    prefix = term_to_slug_prefix(term_name)
    return f"{prefix}-{subject}-{course}-{section}-({crn})"


def parse_title(title: str):
    m = _TITLE_RE.match(title.strip())
    return m.groupdict() if m else None


def output_dir_for(subject: str, term: str) -> str:
    subgroup = "graduate" if GRADUATE_ONLY else "undergraduate"
    return os.path.join(RAW_ROOT, subject, subgroup, term_to_folder(term))


def fetch_json(page, url: str) -> dict:
    """Fetch a JSON API page via the browser context, retrying through transient
    CloudFront/WAF blocks (HTTP 403 + HTML error page) with exponential backoff.

    Between retries we re-seed the session by reloading the library page, which refreshes
    cookies. Raises RuntimeError if still blocked after MAX_RETRIES.
    """
    status, body = None, ""
    for attempt in range(1, MAX_RETRIES + 1):
        res = page.evaluate(
            """async (u) => {
                try {
                    const r = await fetch(u, {headers: {Accept: 'application/json'}});
                    return {status: r.status, body: await r.text()};
                } catch (e) {
                    return {status: -1, body: String(e)};
                }
            }""",
            url,
        )
        status, body = res.get("status"), res.get("body", "")
        if status == 200:
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                pass  # 200 but not JSON — treat as transient and retry
        wait = min(60, 5 * 2 ** (attempt - 1)) + random.uniform(0, 3)
        print(f"    [retry {attempt}/{MAX_RETRIES}] status={status}; backing off {wait:.0f}s…")
        time.sleep(wait)
        try:
            page.goto(f"{BASE}/en-US/syllabus-library", wait_until="networkidle", timeout=45000)
        except Exception:
            pass
    raise RuntimeError(f"Blocked after {MAX_RETRIES} retries (last status={status}): {url}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()

        # ── Phase 1: seed session + scan ────────────────────────────────────
        print("Seeding session…")
        page.goto(f"{BASE}/en-US/syllabus-library", wait_until="networkidle")

        print("Scanning syllabi…")
        candidates = []
        seen_codes: set[str] = set()

        for dept in sorted(DEPARTMENTS):
            print(f"\n[scan] department={dept}")
            pg = 0
            total = None
            dept_matches_before = len(candidates)

            while True:
                url = f"{API}?{PARAMS.format(dept=dept, ps=PAGE_SIZE, pg=pg)}"
                data = fetch_json(page, url)

                items = data.get("items", [])
                if total is None:
                    total = data["pagination"]["total"]
                    pages = -(-total // PAGE_SIZE)
                    print(f"  Total: {total} ({pages} pages)")

                for item in items:
                    title = item.get("title", "")
                    term_name = item.get("term_name", "")
                    code = item.get("code", "")

                    term = term_name.split(" - ")[0] if " - " in term_name else term_name
                    # Handle code-style term names (e.g. "202521" → "Summer 2025")
                    term = _CODE_TO_TERM.get(term, term)
                    if term not in TARGET_TERMS:
                        continue
                    if " - " in term_name and "College Station" not in term_name:
                        continue

                    parsed = parse_title(title)
                    if not parsed:
                        continue
                    if parsed["subject"] not in DEPARTMENTS:
                        continue
                    if GRADUATE_ONLY and int(parsed["course"]) < 600:
                        continue
                    if code in seen_codes:
                        continue
                    seen_codes.add(code)

                    candidates.append({"code": code, "term_name": term_name, "term": term, **parsed})

                fetched = pg * PAGE_SIZE + len(items)
                print(f"  Page {pg}: {fetched}/{total} scanned, {len(candidates) - dept_matches_before} dept matches")

                if fetched >= total or not items:
                    break
                pg += 1
                time.sleep(DELAY)

        print(f"\nFound {len(candidates)} syllabi to download.")

        # ── Phase 2: print each syllabus view page to PDF ───────────────────
        # Group metadata by output directory
        meta_by_dir: dict[str, dict] = {}
        est_bytes = 0.0  # running estimate of cellular data used (for the optional budget)

        for i, c in enumerate(candidates, 1):
            subject = c["subject"]
            course = c["course"]
            section = c["section"]
            crn = c["crn"]
            code = c["code"]
            term_name = c["term_name"]
            term = c["term"]
            term_code = term_to_code(term)

            out_dir = output_dir_for(subject, term)
            os.makedirs(out_dir, exist_ok=True)

            slug = build_slug(term_name, subject, course, section, crn)
            slug_enc = quote(slug, safe="-")
            view_url = f"{BASE}/en-US/doc/{code}/{slug_enc}?mode=view"
            filename = f"{term_code}_{subject}_{course}_{section}_{crn}.pdf"
            out_path = os.path.join(out_dir, filename)

            meta_by_dir.setdefault(out_dir, {})[filename] = {
                "syllabus_url": view_url,
                "doc_id": code,
            }

            if os.path.exists(out_path):
                print(f"[{i}/{len(candidates)}] Skip (exists): {filename}")
                continue

            print(f"[{i}/{len(candidates)}] Printing {filename}...", end=" ", flush=True)
            try:
                page.goto(view_url, wait_until="networkidle", timeout=30000)
                page.pdf(path=out_path, format="Letter", print_background=True)
                size = os.path.getsize(out_path)
                est_bytes += size * NET_FACTOR
                est_mb = est_bytes / 1_000_000
                print(f"OK ({size // 1024}KB)  [~{est_mb:.0f} MB est. data used]")
            except Exception as e:
                print(f"ERROR: {e}")

            if MAX_MB and est_bytes / 1_000_000 >= MAX_MB:
                print(
                    f"\n⛔ Data budget reached (~{est_bytes / 1_000_000:.0f} MB ≥ {MAX_MB:.0f} MB cap). "
                    f"Stopping. Re-run later to resume — already-downloaded files are skipped."
                )
                break

            time.sleep(DELAY)

        browser.close()

    # Write per-directory metadata files
    for out_dir, meta in meta_by_dir.items():
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata: {meta_path} ({len(meta)} entries)")

    print(f"\nDone. Output root: {RAW_ROOT}")
    print(f"Estimated cellular data used this run: ~{est_bytes / 1_000_000:.0f} MB")


if __name__ == "__main__":
    main()
