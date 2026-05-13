"""Ingest Gemini-parsed JSON files into MongoDB Atlas.

Usage:
    python -m tamubot.ingestion.ingest                          # ingest all
    python -m tamubot.ingestion.ingest --department CSCE         # single department
    python -m tamubot.ingestion.ingest --dry-run                 # preview without writing

Requires MONGODB_URI and VOYAGE_API_KEY in .env.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import voyageai
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from tamubot.rag.models import ChunkDoc, CourseDoc
from tamubot.rag.models_v3 import ChunkDocV3, CourseDocV3

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "tamubot")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PARSED_DIR_LEGACY = Path("tamu_data/processed/gemini_parsed")
PARSED_DIR_V3 = Path("tamu_data/processed/v3_step3_flat")
PARSED_DIR_V4 = Path("data/syllabi/silver/06_chunk")

EMBEDDING_MODEL = "voyage-3"
EMBEDDING_DIMS = 1024
EMBED_BATCH_SIZE = 50


def get_db():
    if not MONGODB_URI:
        print("ERROR: MONGODB_URI not set.")
        sys.exit(1)
    return MongoClient(MONGODB_URI)[DB_NAME]


def get_voyage_client():
    if not VOYAGE_API_KEY:
        print("ERROR: VOYAGE_API_KEY not set.")
        sys.exit(1)
    return voyageai.Client(api_key=VOYAGE_API_KEY)


def build_anchor(course_id: str, section: str, term: str, category: str) -> str:
    """Build a contextual anchor string prepended to chunk text before embedding."""
    return f"{course_id} Section {section}, {term} — {category}:"


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()


def parse_json_file(filepath: Path) -> dict | None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: Could not read {filepath.name}: {e}")
        return None


def build_chunk_docs(data: dict, source_file: str) -> list[dict]:
    """Convert parsed JSON into validated chunk documents ready for MongoDB."""
    meta = data.get("course_metadata", {})
    crn = meta.get("crn", "")
    course_id = meta.get("course_id", "")
    section = meta.get("section", "")
    term = meta.get("term", "")
    instructor = meta.get("instructor", {})
    instructor_name = instructor.get("name") if instructor else None

    is_v3 = data.get("pipeline_version") == "v3"

    docs = []
    for i, chunk in enumerate(data.get("chunks", [])):
        category = chunk.get("category") or ("SYLLABUS_V3" if is_v3 else "")
        title = chunk.get("title") or (f"Chunk {i}" if is_v3 else "")
        anchor = build_anchor(course_id, section, term, category)
        validated = ChunkDoc(
            crn=crn,
            chunk_index=i,
            category=category,
            title=title,
            content=chunk.get("content") or "",
            has_table=chunk.get("has_table") or False,
            anchor=anchor,
            course_id=course_id,
            section=section,
            term=term,
            instructor_name=instructor_name,
            source_file=source_file,
        )
        docs.append(validated.model_dump())
    return docs


def build_course_doc(data: dict, source_file: str) -> dict:
    meta = data.get("course_metadata", {})
    chunks = data.get("chunks", [])
    categories = list({c.get("category", "") for c in chunks})
    if data.get("pipeline_version") == "v3":
        categories = ["SYLLABUS_V3"]
    completeness = data.get("completeness_check", {})
    validated = CourseDoc(
        crn=meta.get("crn", ""),
        course_id=meta.get("course_id", ""),
        section=meta.get("section", ""),
        term=meta.get("term", ""),
        instructor=meta.get("instructor"),
        teaching_assistants=meta.get("teaching_assistants") or [],
        meeting_times=meta.get("meeting_times"),
        location=meta.get("location"),
        credit_hours=meta.get("credit_hours"),
        categories_present=categories,
        chunk_count=len(chunks),
        boilerplate_policies=data.get("boilerplate_policies", []),
        missing_sections=completeness.get("missing_sections", []),
        completeness_warnings=completeness.get("warnings", []),
        source_file=source_file,
    )
    return validated.model_dump()


_SEMESTER_MAP = {
    "11": "Spring",
    "21": "Summer I",
    "31": "Summer II",
    "41": "Fall",
    "01": "Spring",
    "02": "Summer",
    "03": "Fall",
}


def _parse_v3_result_filename(stem: str) -> dict:
    """Extract metadata from v3_result filename: {semester}_{dept}_{num}_{section}_{crn}_v{ver}.

    Returns dict with keys: crn, course_id, section, term (empty string on parse failure).
    """
    parts = stem.split("_")
    # Expected: ['202541', 'CSCE', '111', '500', '49744', 'v010']
    if len(parts) < 5:
        return {"crn": "", "course_id": "", "section": "", "term": ""}
    semester = parts[0]  # e.g. '202541'
    dept = parts[1]  # e.g. 'CSCE'
    course_num = parts[2]  # e.g. '111'
    section = parts[3]  # e.g. '500'
    crn = parts[4]  # e.g. '49744'
    year = semester[:4]
    sem_code = semester[4:]
    season = _SEMESTER_MAP.get(sem_code, f"Term{sem_code}")
    term = f"{season} {year}"
    course_id = f"{dept} {course_num}"
    return {"crn": crn, "course_id": course_id, "section": section, "term": term}


def build_chunk_docs_v3_result(
    data: dict,
    source_file: str,
    chunk_tag: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Build ChunkDocV3 docs from v3_result format files.

    v3_result format has top-level course_id/semester/source_file instead of
    course_metadata. CRN, section, term are parsed from source_file filename.
    """
    stem = Path(source_file).stem
    meta = _parse_v3_result_filename(stem)
    crn = meta["crn"]
    course_id = meta["course_id"]
    section = meta["section"]
    term = meta["term"]

    # chunk_size/overlap from file if not provided on CLI
    eff_chunk_size = chunk_size if chunk_size is not None else data.get("chunk_size")
    eff_chunk_overlap = chunk_overlap if chunk_overlap is not None else data.get("overlap")

    docs = []
    for i, chunk in enumerate(data.get("chunks", [])):
        idx = chunk.get("chunk_index", i)
        validated = ChunkDocV3(
            crn=crn,
            chunk_index=idx,
            content=chunk.get("content") or "",
            has_table=chunk.get("has_table") or False,
            course_id=course_id,
            section=section,
            term=term,
            instructor_name=None,
            source_file=source_file,
            chunk_tag=chunk_tag,
            chunk_size=eff_chunk_size,
            chunk_overlap=eff_chunk_overlap,
        )
        docs.append(validated.model_dump())
    return docs


def build_course_doc_v3_result(data: dict, source_file: str) -> dict:
    """Build CourseDocV3 doc from v3_result format files."""
    stem = Path(source_file).stem
    meta = _parse_v3_result_filename(stem)
    chunks = data.get("chunks", [])
    validated = CourseDocV3(
        crn=meta["crn"],
        course_id=meta["course_id"],
        section=meta["section"],
        term=meta["term"],
        chunk_count=len(chunks),
        source_file=source_file,
    )
    return validated.model_dump()


def build_chunk_docs_v3(
    data: dict,
    source_file: str,
    chunk_tag: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Build ChunkDocV3 documents for the chunks_v3 collection."""
    meta = data.get("course_metadata", {})
    crn = str(meta.get("crn") or "")
    course_id = meta.get("course_id", "")
    section = meta.get("section", "")
    term = meta.get("term", "")
    instructor = meta.get("instructor", {})
    instructor_name = instructor.get("name") if instructor else None

    docs = []
    for i, chunk in enumerate(data.get("chunks", [])):
        validated = ChunkDocV3(
            crn=crn,
            chunk_index=i,
            content=chunk.get("content") or "",
            has_table=chunk.get("has_table") or False,
            course_id=course_id,
            section=section,
            term=term,
            instructor_name=instructor_name,
            source_file=source_file,
            chunk_tag=chunk_tag,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs.append(validated.model_dump())
    return docs


def build_course_doc_v3(data: dict, source_file: str) -> dict:
    """Build a CourseDocV3 document for the courses_v3 collection."""
    meta = data.get("course_metadata", {})
    chunks = data.get("chunks", [])
    validated = CourseDocV3(
        crn=str(meta.get("crn") or ""),
        course_id=meta.get("course_id", ""),
        section=meta.get("section", ""),
        term=meta.get("term", ""),
        instructor=meta.get("instructor"),
        teaching_assistants=meta.get("teaching_assistants") or [],
        meeting_times=meta.get("meeting_times"),
        location=meta.get("location"),
        credit_hours=meta.get("credit_hours"),
        chunk_count=len(chunks),
        syllabus_url=meta.get("syllabus_url"),
        doc_id=meta.get("doc_id"),
        source_file=source_file,
    )
    return validated.model_dump()


def _term_from_filename(stem: str) -> str:
    """Parse term from filename semester code as fallback for malformed metadata.

    Filename pattern: {semester}_{dept}_{num}_{section}_{crn}[_HP][_vNNN]
    Semester: YYYYSS where SS is 11=Spring, 21=Summer, 31=Summer, 41=Fall
    """
    parts = stem.split("_")
    if not parts:
        return ""
    semester = parts[0]
    if len(semester) < 5:
        return ""
    year = semester[:4]
    sem_code = semester[4:]
    season = _SEMESTER_MAP.get(sem_code, f"Term{sem_code}")
    return f"{season} {year}"


def _normalize_tas(raw: list | None) -> list[str]:
    """Normalize teaching_assistants to list of strings.

    V4 metadata enrichment may return TAs as dicts with name/email or as plain strings.
    """
    if not raw:
        return []
    result = []
    for ta in raw:
        if isinstance(ta, str):
            result.append(ta)
        elif isinstance(ta, dict) and ta.get("name"):
            result.append(ta["name"])
    return result


def build_anchor_v4(course_id: str, section: str, term: str, header_path: str) -> str:
    """Build contextual anchor from header_path for v4 chunks."""
    leaf = header_path.rsplit(" > ", 1)[-1] if header_path else "General"
    return f"{course_id} Section {section}, {term} — {leaf}:"


def build_chunk_docs_v4(data: dict, source_file: str) -> list[dict]:
    """Build ChunkDocV4 documents from v4 pipeline output."""
    from tamubot.rag.models_v4 import ChunkDocV4

    meta = data.get("course_metadata", {})
    stem = Path(source_file).stem
    crn = str(meta.get("crn") or "")
    course_id = meta.get("course_id", "")
    section = meta.get("section", "")
    term = meta.get("term", "")
    if "?" in term or not term:
        term = _term_from_filename(stem)
    instructor = meta.get("instructor", {})
    instructor_name = instructor.get("name") if instructor else None
    source = data.get("source")

    docs = []
    for i, chunk in enumerate(data.get("chunks", [])):
        idx = chunk.get("chunk_index", i)
        header_path = chunk.get("header_path", "")
        anchor = build_anchor_v4(course_id, section, term, header_path)
        validated = ChunkDocV4(
            crn=crn,
            chunk_index=idx,
            content=chunk.get("content") or "",
            has_table=chunk.get("has_table") or False,
            course_id=course_id,
            section=section,
            term=term,
            instructor_name=instructor_name,
            header_path=header_path or None,
            token_count=chunk.get("token_count"),
            flags=chunk.get("flags") or [],
            split_reason=chunk.get("split_reason"),
            page=chunk.get("page"),
            source=source,
            anchor=anchor,
            source_file=source_file,
        )
        docs.append(validated.model_dump())
    return docs


def build_course_doc_v4(data: dict, source_file: str) -> dict:
    """Build CourseDocV4 document from v4 pipeline output."""
    from tamubot.rag.models_v4 import CourseDocV4

    meta = data.get("course_metadata", {})
    stem = Path(source_file).stem
    chunks = data.get("chunks", [])
    term = meta.get("term", "")
    if "?" in term or not term:
        term = _term_from_filename(stem)
    validated = CourseDocV4(
        crn=str(meta.get("crn") or ""),
        course_id=meta.get("course_id", ""),
        section=meta.get("section", ""),
        term=term,
        instructor=meta.get("instructor"),
        teaching_assistants=_normalize_tas(meta.get("teaching_assistants")),
        meeting_times=meta.get("meeting_times"),
        location=meta.get("location"),
        credit_hours=meta.get("credit_hours"),
        chunk_count=len(chunks),
        syllabus_url=meta.get("syllabus_url"),
        source=data.get("source"),
        course_type=data.get("course_type"),
        format=meta.get("format"),
        prerequisites=meta.get("prerequisites"),
        course_summary=data.get("course_summary"),
        chunk_config=data.get("chunk_config"),
        source_file=source_file,
    )
    return validated.model_dump()


def build_policy_ops(db, data: dict) -> list[UpdateOne]:
    """Build upsert operations for boilerplate policies."""
    crn = data.get("course_metadata", {}).get("crn", "")
    ops = []
    for name in data.get("boilerplate_policies", []):
        h = sha256_hash(name)
        ops.append(
            UpdateOne(
                {"policy_hash": h},
                {
                    "$set": {"policy_name": name, "ingested_at": datetime.utcnow()},
                    "$setOnInsert": {"policy_hash": h, "full_text": None},
                    "$addToSet": {"courses_referencing": crn},
                },
                upsert=True,
            )
        )
    return ops


def embed_chunks(voyage: voyageai.Client, chunk_docs: list[dict]) -> list[dict]:
    """Embed chunk docs in batches using Voyage AI. Mutates docs in-place."""
    texts = [(doc["anchor"] + " " + doc["content"]) if "anchor" in doc else doc["content"] for doc in chunk_docs]

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        retries = 0
        while True:
            try:
                result = voyage.embed(batch, model=EMBEDDING_MODEL, input_type="document")
                for j, emb in enumerate(result.embeddings):
                    chunk_docs[start + j]["embedding"] = emb
                break
            except Exception as e:
                retries += 1
                if retries > 3:
                    print(f"  ERROR: Embedding failed after 3 retries: {e}")
                    raise
                wait = 2**retries
                print(f"  WARN: Embedding error ({e}), retrying in {wait}s...")
                time.sleep(wait)

    return chunk_docs


def upsert_chunks(db, chunk_docs: list[dict]):
    """Idempotent upsert of chunk documents keyed by (crn, chunk_index)."""
    if not chunk_docs:
        return 0
    ops = [
        UpdateOne(
            {"crn": doc["crn"], "chunk_index": doc["chunk_index"]},
            {"$set": doc},
            upsert=True,
        )
        for doc in chunk_docs
    ]
    result = db["chunks"].bulk_write(ops, ordered=False)
    return result.upserted_count + result.modified_count


def upsert_course(db, course_doc: dict):
    db["courses"].update_one(
        {"crn": course_doc["crn"]},
        {"$set": course_doc},
        upsert=True,
    )


def _crn_from_filename(filepath: Path) -> str | None:
    """Extract CRN from filename pattern like 202611_CSCE_670_600_46627.json.

    Handles suffixes: _vNNN (v3), _HP (v4 Howdy Portal).
    """
    parts = filepath.stem.split("_")
    # For V3 filenames like ..._v001.json, crn is before _vNNN
    if parts and parts[-1].startswith("v") and parts[-1][1:].isdigit():
        if len(parts) >= 2 and parts[-2].isdigit():
            return parts[-2]
    # For V4 filenames with _HP suffix, crn is before _HP
    if parts and parts[-1] == "HP":
        if len(parts) >= 2 and parts[-2].isdigit():
            return parts[-2]
    # For legacy/v4 filenames ending in CRN
    if parts and parts[-1].isdigit():
        return parts[-1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Ingest parsed syllabi into MongoDB Atlas")
    parser.add_argument("--department", type=str, help="Filter by department prefix (e.g. CSCE)")
    parser.add_argument(
        "--crns-file",
        type=str,
        help="JSON file with 'crns' list — ingest only those CRNs (e.g. tamu_data/evals/eval_corpus.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument("--v3", action="store_true", help="Ingest from V3 flat JSONs (v3_step3_flat format)")
    parser.add_argument(
        "--v3-result",
        action="store_true",
        help="Ingest from v3_result chunker output format (top-level course_id/semester, CRN from filename)",
    )
    parser.add_argument(
        "--v4",
        action="store_true",
        help="Ingest from v4 Docling pipeline output (data/syllabi/silver/06_chunk/)",
    )
    parser.add_argument(
        "--source-dir", type=str, help="Override source directory (default: tamu_data/processed/v3_step3_flat)"
    )
    parser.add_argument(
        "--chunks-collection", type=str, help="Target MongoDB collection for chunks (default: chunks_v3 with --v3)"
    )
    parser.add_argument(
        "--courses-collection", type=str, help="Target MongoDB collection for courses (default: courses_v3 with --v3)"
    )
    parser.add_argument(
        "--chunk-tag", type=str, help="Strategy tag stored on each chunk doc (e.g. '300t_50o', 'semantic_v1')"
    )
    parser.add_argument("--chunk-size", type=int, help="Chunk token size stored on each chunk doc (for reporting)")
    parser.add_argument(
        "--chunk-overlap", type=int, help="Chunk overlap token size stored on each chunk doc (for reporting)"
    )
    args = parser.parse_args()

    if args.source_dir:
        parsed_dir = Path(args.source_dir)
    elif args.v4:
        parsed_dir = PARSED_DIR_V4
    elif args.v3 or getattr(args, "v3_result", False):
        parsed_dir = PARSED_DIR_V3
    else:
        parsed_dir = PARSED_DIR_LEGACY
    if not parsed_dir.exists():
        print(f"ERROR: Parsed directory not found: {parsed_dir}")
        sys.exit(1)

    json_files = sorted(parsed_dir.glob("*.json"))
    # Filter out _annotations.json and _run_meta_vNNN.json
    json_files = [f for f in json_files if not f.name.startswith("_")]

    if args.department:
        dept = args.department.upper()
        json_files = [f for f in json_files if f"_{dept}_" in f.name]

    if args.crns_file:
        with open(args.crns_file) as f:
            corpus = json.load(f)
        target_crns = {str(c) for c in corpus.get("crns", [])}
        if not target_crns:
            print(f"ERROR: No CRNs found in {args.crns_file}")
            sys.exit(1)
        json_files = [f for f in json_files if _crn_from_filename(f) in target_crns]
        print(f"  --crns-file: filtered to {len(json_files)} files matching {len(target_crns)} corpus CRNs")

    print(f"Found {len(json_files)} JSON files to ingest from {parsed_dir}")
    if not json_files:
        return

    is_v3 = args.v3 or getattr(args, "v3_result", False)
    if args.v4:
        chunks_col = args.chunks_collection or "chunks_v4"
        courses_col = args.courses_collection or "courses_v4"
    elif is_v3:
        chunks_col = args.chunks_collection or "chunks_v3"
        courses_col = args.courses_collection or "courses_v3"
    else:
        chunks_col = args.chunks_collection or "chunks"
        courses_col = args.courses_collection or "courses"

    if args.dry_run:
        print("DRY RUN — no database writes will occur")
        for f in json_files:
            data = parse_json_file(f)
            if data:
                if "error" in data:
                    print(f"  {f.name}: SKIP (error file)")
                    continue
                if args.v4:
                    chunks = build_chunk_docs_v4(data, f.name)
                elif getattr(args, "v3_result", False):
                    chunks = build_chunk_docs_v3_result(
                        data,
                        f.name,
                        chunk_tag=args.chunk_tag,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap,
                    )
                elif args.v3:
                    chunks = build_chunk_docs_v3(
                        data,
                        f.name,
                        chunk_tag=args.chunk_tag,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap,
                    )
                else:
                    chunks = build_chunk_docs(data, f.name)
                print(f"  {f.name}: {len(chunks)} chunks")
        return

    db = get_db()
    voyage = get_voyage_client()

    total_chunks = 0
    total_files = 0
    errors = []

    for i, filepath in enumerate(json_files):
        data = parse_json_file(filepath)
        if data is None:
            errors.append(filepath.name)
            continue

        if "error" in data:
            print(f"  SKIP: {filepath.name} — previously failed")
            continue

        try:
            # Build documents
            if args.v4:
                chunk_docs = build_chunk_docs_v4(data, filepath.name)
                course_doc = build_course_doc_v4(data, filepath.name)
                policy_ops = []
            elif getattr(args, "v3_result", False):
                chunk_docs = build_chunk_docs_v3_result(
                    data,
                    filepath.name,
                    chunk_tag=args.chunk_tag,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
                course_doc = build_course_doc_v3_result(data, filepath.name)
                policy_ops = []
            elif args.v3:
                chunk_docs = build_chunk_docs_v3(
                    data,
                    filepath.name,
                    chunk_tag=args.chunk_tag,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
                course_doc = build_course_doc_v3(data, filepath.name)
                policy_ops = []
            else:
                chunk_docs = build_chunk_docs(data, filepath.name)
                course_doc = build_course_doc(data, filepath.name)
                policy_ops = build_policy_ops(db, data)

            # Embed
            if chunk_docs:
                embed_chunks(voyage, chunk_docs)

            # Write to MongoDB
            db[courses_col].update_one({"crn": course_doc["crn"]}, {"$set": course_doc}, upsert=True)
            if chunk_docs:
                ops = []
                for doc in chunk_docs:
                    filt = {"crn": doc["crn"], "chunk_index": doc["chunk_index"]}
                    if args.chunk_tag:
                        filt["chunk_tag"] = doc.get("chunk_tag", args.chunk_tag)
                    ops.append(UpdateOne(filt, {"$set": doc}, upsert=True))
                db[chunks_col].bulk_write(ops, ordered=False)
            if policy_ops:
                db["policies"].bulk_write(policy_ops, ordered=False)

            total_chunks += len(chunk_docs)
            total_files += 1
            print(f"  [{i + 1}/{len(json_files)}] {filepath.name}: {len(chunk_docs)} chunks ingested")

        except Exception as e:
            errors.append(filepath.name)
            print(f"  [{i + 1}/{len(json_files)}] ERROR {filepath.name}: {e}")

    print(f"\nIngestion complete: {total_files} files, {total_chunks} chunks")
    print(f"  Courses ({courses_col}): {db[courses_col].count_documents({})}")
    print(f"  Chunks  ({chunks_col}):  {db[chunks_col].count_documents({})}")
    if not (args.v3 or args.v4):
        print(f"  Policies: {db['policies'].count_documents({})}")
    if errors:
        print(f"  Errors ({len(errors)}): {errors}")


if __name__ == "__main__":
    main()
