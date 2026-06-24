"""Backfill the Postgres retrieval backend from the existing sources.

Phase 1 ETL: stream the live Atlas collections plus the on-disk structured JSON
into the Option-2 Postgres schema, **copying embeddings as-is** (same Voyage
``voyage-3`` model — no re-embedding).

Sources → tables:
  * ``chunks_v4``                        → ``documents`` (one per source_file) + ``chunks``
  * ``courses_v4``                       → ``courses`` + ``sections`` + ``instructors``
  * ``courses_v4.course_summary``        → ``course_topics``         (parse the "Topics:" line)
  * ``silver/05_structured/<stem>.json`` → ``section_assessments`` / ``section_grade_cutoffs`` / ``learning_outcomes``

Every load is idempotent: documents/instructors/courses/sections/course_topics
upsert ``ON CONFLICT``; chunks upsert on the pipeline key
``(source_file, chunk_index, chunk_tag)``; the structured child tables are
replace-by-CRN (delete-then-insert) since they have no natural key.

Run inside the dev container (host Python can't load the stack):
    python -m tamubot.ingestion.postgres.backfill            # full backfill
    python -m tamubot.ingestion.postgres.backfill --dry-run  # counts only, no writes
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Iterable

from tamubot.core import config
from tamubot.ingestion.pipeline_v5.util import DATA_ROOT
from tamubot.ingestion.postgres.pool import get_pool

# chunks_v4 embeddings were produced by Voyage voyage-3 (see rag/tools/voyage.py).
EMBEDDING_MODEL = "voyage-3"


# ─────────────────────────────────────────────────────────────────────────────
# Pure transforms (no DB / no network — unit-tested directly)
# ─────────────────────────────────────────────────────────────────────────────


def dept_of(course_id: str | None) -> str | None:
    """'ISEN 625' → 'ISEN'. None/blank → None."""
    if not course_id:
        return None
    head = course_id.strip().split()
    return head[0].upper() if head else None


def stem_of(source_file: str | None) -> str | None:
    """The pipeline stem is the source_file minus its '.json' suffix.

    Chunks/courses store ``source_file`` WITH the ``.json`` suffix
    ('202641_ISEN_625_601_52030.json'); the structured JSON on disk is named by
    the bare stem ('202641_ISEN_625_601_52030.json' → stem
    '202641_ISEN_625_601_52030'). This is the join key between them.
    """
    if not source_file:
        return None
    return source_file[:-5] if source_file.endswith(".json") else source_file


def parse_topics(course_summary: str | None) -> list[str]:
    """Extract the comma-separated terms from the course_summary 'Topics:' line.

    Mirrors the format written by ``filters/metadata_enrichment.py`` — a single
    line ``Topics: a, b, c``. Returns an order-preserving, de-duplicated list;
    empty when there is no Topics line.
    """
    if not course_summary:
        return []
    for line in course_summary.split("\n"):
        if line.strip().startswith("Topics:"):
            raw = line.split(":", 1)[1]
            seen: set[str] = set()
            out: list[str] = []
            for term in raw.split(","):
                t = term.strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower())
                    out.append(t)
            return out
    return []


def chunk_row(chunk: dict, *, embedding_model: str = EMBEDDING_MODEL) -> dict[str, Any]:
    """Map a chunks_v4 Mongo doc to a ``chunks`` insert row (sans document_id).

    ``document_id`` is resolved by the loader from the source_file→id map.
    Boilerplate/duplicate fields default to False so pre-v6 chunks (which lack
    them) behave as content — matching the Mongo ``$ne: True`` semantics.
    """
    return {
        "doc_type": "syllabus",
        "source_file": chunk["source_file"],
        "chunk_index": chunk["chunk_index"],
        "chunk_tag": chunk.get("chunk_tag") or "semantic",
        "course_id": chunk.get("course_id"),
        "crn": chunk.get("crn"),
        "term": chunk.get("term"),
        "section": chunk.get("section"),
        "instructor_name": chunk.get("instructor_name"),
        "source": chunk.get("source"),
        "content": chunk.get("content") or "",
        "header_path": chunk.get("header_path"),
        "anchor": chunk.get("anchor"),
        "page": chunk.get("page"),
        "token_count": chunk.get("token_count"),
        "has_table": bool(chunk.get("has_table", False)),
        "split_reason": chunk.get("split_reason"),
        "flags": chunk.get("flags") or [],
        "is_boilerplate": bool(chunk.get("is_boilerplate", False)),
        "boilerplate_cluster": chunk.get("boilerplate_cluster"),
        "cluster_confidence": chunk.get("cluster_confidence"),
        "is_duplicate": bool(chunk.get("is_duplicate", False)),
        "embedding": chunk.get("embedding"),
        "embedding_model": embedding_model if chunk.get("embedding") else None,
    }


def structured_rows(extract: dict) -> dict:
    """Split a SyllabusExtract dict into child-table rows + section policy scalars.

    Returns ``{"assessments", "cutoffs", "outcomes", "meetings",
    "prerequisites", "attendance_policy", "academic_integrity_policy"}``. The
    loader attaches the CRN. Empty/None inputs yield empty lists / None.
    """
    assessments = [
        {"component": a.get("component"), "weight_pct": a.get("weight_pct")}
        for a in (extract.get("assessment_weights") or [])
    ]
    cutoffs = [
        {"grade": c.get("grade"), "min_percent": c.get("min_percent")}
        for c in (extract.get("letter_grade_cutoffs") or [])
    ]
    outcomes = [
        {"ordinal": i, "text": text}
        for i, text in enumerate(extract.get("learning_outcomes") or [])
        if text
    ]
    meetings = [
        {"ordinal": i, "day": m.get("day"), "time": m.get("time"), "location": m.get("location")}
        for i, m in enumerate(extract.get("meeting_schedule") or [])
    ]
    prerequisites = [
        {"ordinal": i, "text": p}
        for i, p in enumerate(extract.get("prerequisites") or [])
        if p
    ]
    return {
        "assessments": assessments,
        "cutoffs": cutoffs,
        "outcomes": outcomes,
        "meetings": meetings,
        "prerequisites": prerequisites,
        "attendance_policy": extract.get("attendance_policy"),
        "academic_integrity_policy": extract.get("academic_integrity_policy"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Source readers
# ─────────────────────────────────────────────────────────────────────────────


def _load_mongo() -> tuple[list[dict], list[dict]]:
    """Read every chunks_v4 + courses_v4 doc from Atlas into memory."""
    from tamubot.rag.tools.mongo import get_db

    db = get_db()
    chunks = list(db["chunks_v4"].find({}))
    courses = list(db["courses_v4"].find({}, {"_id": 0}))
    return chunks, courses


def _load_structured() -> dict[str, dict]:
    """Read every silver/05_structured/<stem>.json keyed by stem."""
    pattern = str(DATA_ROOT / "*" / "v6b" / "silver" / "05_structured" / "*.json")
    out: dict[str, dict] = {}
    for path in glob.glob(pattern):
        stem = Path(path).stem
        try:
            out[stem] = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def _document_rows(chunks: Iterable[dict]) -> list[dict]:
    """One document per distinct source_file, enriched from its first chunk.

    Carries ``crn`` so each document links N:1 to its section (a CRN with both a
    Howdy Portal and a Simple Syllabus copy yields two documents, same CRN).
    """
    seen: dict[str, dict] = {}
    for c in chunks:
        sf = c.get("source_file")
        if not sf or sf in seen:
            continue
        cid = c.get("course_id")
        section = c.get("section")
        title = " ".join(p for p in (cid, section) if p) or sf
        seen[sf] = {
            "doc_type": "syllabus",
            "title": title,
            "source": c.get("source"),
            "source_file": sf,
            "crn": c.get("crn"),
            "dept": dept_of(cid),
        }
    return list(seen.values())


def backfill(*, dry_run: bool = False, log=print) -> dict[str, int]:
    """Run the full backfill. Returns a per-table written-row count."""
    chunks, courses = _load_mongo()
    structured = _load_structured()
    log(f"loaded {len(chunks)} chunks, {len(courses)} courses, {len(structured)} structured JSON")

    documents = _document_rows(chunks)
    counts = {
        "instructors": 0,
        "courses": 0,
        "sections": len(courses),
        "documents": len(documents),
        "chunks": len(chunks),
        "section_topics": 0,
        "section_assessments": 0,
        "section_grade_cutoffs": 0,
        "learning_outcomes": 0,
        "section_meetings": 0,
        "section_prerequisites": 0,
        "section_policies": 0,
    }
    if dry_run:
        log("dry-run: no writes")
        return counts

    pool = get_pool()
    with pool.connection() as conn:
        with conn.transaction():
            # FK order: instructors/courses → sections → documents → chunks → section children.
            instr_id = _upsert_instructors(conn, courses)
            counts["instructors"] = len(instr_id)
            counts["courses"] = _upsert_courses(conn, courses)
            _upsert_sections(conn, courses, instr_id)
            doc_id = _upsert_documents(conn, documents)
            _upsert_chunks(conn, chunks, doc_id)
            counts["section_topics"] = _upsert_section_topics(conn, courses)
            s = _load_structured_rows(conn, courses, structured)
            counts.update(s)
    log(f"backfill complete: {counts}")
    return counts


def _upsert_documents(conn, documents: list[dict]) -> dict[str, int]:
    for d in documents:
        conn.execute(
            """
            INSERT INTO documents (doc_type, title, source, source_file, crn, dept)
            VALUES (%(doc_type)s, %(title)s, %(source)s, %(source_file)s, %(crn)s, %(dept)s)
            ON CONFLICT (source_file, doc_type) DO UPDATE
              SET title = EXCLUDED.title, source = EXCLUDED.source,
                  crn = EXCLUDED.crn, dept = EXCLUDED.dept
            """,
            d,
        )
    rows = conn.execute(
        "SELECT source_file, id FROM documents WHERE doc_type = 'syllabus'"
    ).fetchall()
    return {sf: i for sf, i in rows}


def _instructor_key(course: dict) -> tuple[str, str | None] | None:
    instr = course.get("instructor") or {}
    name = (instr.get("name") or "").strip()
    if not name:
        return None
    return name, dept_of(course.get("course_id"))


def _upsert_instructors(conn, courses: list[dict]) -> dict[tuple[str, str | None], int]:
    for course in courses:
        key = _instructor_key(course)
        if key is None:
            continue
        name, dept = key
        instr = course.get("instructor") or {}
        conn.execute(
            """
            INSERT INTO instructors (name, email, dept)
            VALUES (%s, %s, %s)
            ON CONFLICT (name, dept) DO UPDATE SET email = COALESCE(EXCLUDED.email, instructors.email)
            """,
            (name, instr.get("email"), dept),
        )
    rows = conn.execute("SELECT name, dept, id FROM instructors").fetchall()
    return {(name, dept): i for name, dept, i in rows}


def _upsert_courses(conn, courses: list[dict]) -> int:
    seen: set[str] = set()
    for course in courses:
        cid = course.get("course_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        conn.execute(
            """
            INSERT INTO courses (course_id, dept, title, prerequisites_text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (course_id) DO UPDATE
              SET dept = EXCLUDED.dept,
                  prerequisites_text = COALESCE(EXCLUDED.prerequisites_text, courses.prerequisites_text)
            """,
            (cid, dept_of(cid), None, course.get("prerequisites")),
        )
    return len(seen)


def _upsert_sections(conn, courses, instr_id) -> None:
    for course in courses:
        crn = course.get("crn")
        if not crn:
            continue
        key = _instructor_key(course)
        iid = instr_id.get(key) if key else None
        conn.execute(
            """
            INSERT INTO sections (crn, course_id, section, term, instructor_id,
                                  meeting_times, location, format, credit_hours, course_type,
                                  syllabus_url, source, course_summary)
            VALUES (%(crn)s, %(course_id)s, %(section)s, %(term)s, %(instructor_id)s,
                    %(meeting_times)s, %(location)s, %(format)s, %(credit_hours)s, %(course_type)s,
                    %(syllabus_url)s, %(source)s, %(course_summary)s)
            ON CONFLICT (crn) DO UPDATE SET
                course_id = EXCLUDED.course_id, section = EXCLUDED.section,
                term = EXCLUDED.term, instructor_id = EXCLUDED.instructor_id,
                meeting_times = EXCLUDED.meeting_times, location = EXCLUDED.location,
                format = EXCLUDED.format, credit_hours = EXCLUDED.credit_hours,
                course_type = EXCLUDED.course_type, syllabus_url = EXCLUDED.syllabus_url,
                source = EXCLUDED.source, course_summary = EXCLUDED.course_summary
            """,
            {
                "crn": crn,
                "course_id": course.get("course_id"),
                "section": course.get("section"),
                "term": course.get("term"),
                "instructor_id": iid,
                "meeting_times": course.get("meeting_times"),
                "location": course.get("location"),
                "format": course.get("format"),
                "credit_hours": course.get("credit_hours"),
                "course_type": course.get("course_type"),
                "syllabus_url": course.get("syllabus_url"),
                "source": course.get("source"),
                "course_summary": course.get("course_summary"),
            },
        )


def _upsert_section_topics(conn, courses: list[dict]) -> int:
    """Topics parsed from each section's course_summary, stored at CRN grain.

    The course-level ``course_topics`` view aggregates these for list_courses.
    """
    written = 0
    for course in courses:
        crn = course.get("crn")
        if not crn:
            continue
        for topic in parse_topics(course.get("course_summary")):
            conn.execute(
                """
                INSERT INTO section_topics (crn, topic) VALUES (%s, %s)
                ON CONFLICT (crn, topic) DO NOTHING
                """,
                (crn, topic),
            )
            written += 1
    return written


def _upsert_chunks(conn, chunks: list[dict], doc_id: dict[str, int]) -> None:
    for c in chunks:
        row = chunk_row(c)
        row["document_id"] = doc_id.get(row["source_file"])
        # flags is a list[str]; psycopg adapts it to the text[] column directly.
        conn.execute(
            """
            INSERT INTO chunks (
                document_id, doc_type, source_file, chunk_index, chunk_tag,
                course_id, crn, term, section, instructor_name, source,
                content, header_path, anchor, page, token_count, has_table,
                split_reason, flags, is_boilerplate, boilerplate_cluster,
                cluster_confidence, is_duplicate, embedding, embedding_model)
            VALUES (
                %(document_id)s, %(doc_type)s, %(source_file)s, %(chunk_index)s, %(chunk_tag)s,
                %(course_id)s, %(crn)s, %(term)s, %(section)s, %(instructor_name)s, %(source)s,
                %(content)s, %(header_path)s, %(anchor)s, %(page)s, %(token_count)s, %(has_table)s,
                %(split_reason)s, %(flags)s, %(is_boilerplate)s, %(boilerplate_cluster)s,
                %(cluster_confidence)s, %(is_duplicate)s, %(embedding)s, %(embedding_model)s)
            ON CONFLICT (source_file, chunk_index, chunk_tag) DO UPDATE SET
                document_id = EXCLUDED.document_id, content = EXCLUDED.content,
                course_id = EXCLUDED.course_id, crn = EXCLUDED.crn, term = EXCLUDED.term,
                section = EXCLUDED.section, instructor_name = EXCLUDED.instructor_name,
                source = EXCLUDED.source,
                header_path = EXCLUDED.header_path, anchor = EXCLUDED.anchor, page = EXCLUDED.page,
                token_count = EXCLUDED.token_count, has_table = EXCLUDED.has_table,
                split_reason = EXCLUDED.split_reason, flags = EXCLUDED.flags,
                is_boilerplate = EXCLUDED.is_boilerplate, boilerplate_cluster = EXCLUDED.boilerplate_cluster,
                cluster_confidence = EXCLUDED.cluster_confidence, is_duplicate = EXCLUDED.is_duplicate,
                embedding = EXCLUDED.embedding, embedding_model = EXCLUDED.embedding_model
            """,
            row,
        )


def _load_structured_rows(conn, courses: list[dict], structured: dict[str, dict]) -> dict[str, int]:
    """Match each structured stem → CRN and replace its child rows (delete+insert)."""
    stem_to_crn: dict[str, str] = {}
    for course in courses:
        stem = stem_of(course.get("source_file"))
        if stem and course.get("crn"):
            stem_to_crn[stem] = course["crn"]

    counts = {
        "section_assessments": 0,
        "section_grade_cutoffs": 0,
        "learning_outcomes": 0,
        "section_meetings": 0,
        "section_prerequisites": 0,
        "section_policies": 0,
    }
    for stem, extract in structured.items():
        crn = stem_to_crn.get(stem)
        if not crn:
            continue  # structured doc with no matching section in courses_v4
        rows = structured_rows(extract)
        # Replace-by-CRN: child tables have no natural key, so delete then insert.
        for table in (
            "section_assessments",
            "section_grade_cutoffs",
            "learning_outcomes",
            "section_meetings",
            "section_prerequisites",
        ):
            conn.execute(f"DELETE FROM {table} WHERE crn = %s", (crn,))
        for a in rows["assessments"]:
            conn.execute(
                "INSERT INTO section_assessments (crn, component, weight_pct) VALUES (%s, %s, %s)",
                (crn, a["component"], a["weight_pct"]),
            )
            counts["section_assessments"] += 1
        for c in rows["cutoffs"]:
            conn.execute(
                "INSERT INTO section_grade_cutoffs (crn, grade, min_percent) VALUES (%s, %s, %s)",
                (crn, c["grade"], c["min_percent"]),
            )
            counts["section_grade_cutoffs"] += 1
        for o in rows["outcomes"]:
            conn.execute(
                "INSERT INTO learning_outcomes (crn, ordinal, text) VALUES (%s, %s, %s)",
                (crn, o["ordinal"], o["text"]),
            )
            counts["learning_outcomes"] += 1
        for m in rows["meetings"]:
            conn.execute(
                "INSERT INTO section_meetings (crn, ordinal, day, time, location) "
                "VALUES (%s, %s, %s, %s, %s)",
                (crn, m["ordinal"], m["day"], m["time"], m["location"]),
            )
            counts["section_meetings"] += 1
        for p in rows["prerequisites"]:
            conn.execute(
                "INSERT INTO section_prerequisites (crn, ordinal, text) VALUES (%s, %s, %s)",
                (crn, p["ordinal"], p["text"]),
            )
            counts["section_prerequisites"] += 1
        # Section policy paragraphs (1:1) — UPDATE the existing section row.
        if rows["attendance_policy"] or rows["academic_integrity_policy"]:
            conn.execute(
                "UPDATE sections SET attendance_policy = %s, academic_integrity_policy = %s "
                "WHERE crn = %s",
                (rows["attendance_policy"], rows["academic_integrity_policy"], crn),
            )
            counts["section_policies"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Postgres from Atlas + structured JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Count source rows; write nothing.")
    args = parser.parse_args()
    try:
        backfill(dry_run=args.dry_run)
    finally:
        if not args.dry_run:
            # Close the pool's worker threads before interpreter shutdown to avoid
            # a PythonFinalizationError from the pool's __del__ joining threads late.
            get_pool().close()


if __name__ == "__main__":
    main()
