"""Create MongoDB Atlas indexes for the three collections.

Usage:
    python -m tamubot.ingestion.setup_atlas

Requires MONGODB_URI in .env (or environment).
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "tamubot")


def get_db():
    if not MONGODB_URI:
        print("ERROR: MONGODB_URI not set in environment.")
        sys.exit(1)
    client = MongoClient(MONGODB_URI)
    return client[DB_NAME]


def setup_standard_indexes(db):
    """Create standard MongoDB indexes for query filtering."""
    chunks = db["chunks"]
    chunks.create_index("crn", name="idx_crn")
    chunks.create_index("course_id", name="idx_course_id")
    chunks.create_index("category", name="idx_category")
    chunks.create_index(
        [("course_id", 1), ("category", 1)],
        name="idx_course_category",
    )
    chunks.create_index(
        [("crn", 1), ("chunk_index", 1)],
        unique=True,
        name="idx_crn_chunk_unique",
    )
    print("  [chunks] standard indexes created")

    courses = db["courses"]
    courses.create_index("crn", unique=True, name="idx_crn")
    courses.create_index("course_id", name="idx_course_id")
    courses.create_index("term", name="idx_term")
    print("  [courses] standard indexes created")

    policies = db["policies"]
    policies.create_index("policy_hash", unique=True, name="idx_policy_hash")
    policies.create_index("policy_name", name="idx_policy_name")
    print("  [policies] standard indexes created")


def setup_search_indexes(db):
    """Create Atlas Search and Vector Search indexes.

    These use the Atlas Search Index API (requires M10+ cluster or Atlas free
    tier with search enabled).
    """
    chunks = db["chunks"]

    # Vector search index for embeddings (Voyage AI voyage-3: 1024 dims)
    vector_idx = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "course_id"},
                {"type": "filter", "path": "category"},
                {"type": "filter", "path": "term"},
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )

    # Full-text search index for keyword/BM25 retrieval
    text_idx = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    "title": {"type": "string", "analyzer": "lucene.standard"},
                    "course_id": {"type": "token"},
                    "category": {"type": "token"},
                    "instructor_name": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
        name="text_index",
        type="search",
    )

    existing = [idx["name"] for idx in chunks.list_search_indexes()]

    created = []
    indexes = {"vector_index": vector_idx, "text_index": text_idx}
    for name, idx in indexes.items():
        if name in existing:
            print(f"  [chunks] search index '{name}' already exists — skipping")
        else:
            chunks.create_search_index(idx)
            created.append(name)
            print(f"  [chunks] search index '{name}' created (may take a few minutes to build)")

    return created


def setup_standard_indexes_v2(db):
    """Create standard MongoDB indexes for the V2 collections."""
    chunks_v2 = db["chunks_v2"]
    chunks_v2.create_index("crn", name="idx_crn")
    chunks_v2.create_index("course_id", name="idx_course_id")
    chunks_v2.create_index("term", name="idx_term")
    chunks_v2.create_index("header_level", name="idx_header_level")
    chunks_v2.create_index(
        [("crn", 1), ("chunk_index", 1)],
        unique=True,
        name="idx_crn_chunk_unique",
    )
    print("  [chunks_v2] standard indexes created")

    courses_v2 = db["courses_v2"]
    courses_v2.create_index("crn", unique=True, name="idx_crn")
    courses_v2.create_index("course_id", name="idx_course_id")
    courses_v2.create_index("term", name="idx_term")
    print("  [courses_v2] standard indexes created")


def setup_search_indexes_v2(db):
    """Create Atlas Search and Vector Search indexes for V2 collections."""
    chunks_v2 = db["chunks_v2"]

    vector_idx = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "course_id"},
                {"type": "filter", "path": "term"},
            ]
        },
        name="vector_index_v2",
        type="vectorSearch",
    )

    text_idx = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    "header_text": {"type": "string", "analyzer": "lucene.standard"},
                    "course_id": {"type": "token"},
                    "term": {"type": "token"},
                    "instructor_name": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
        name="text_index_v2",
        type="search",
    )

    existing = [idx["name"] for idx in chunks_v2.list_search_indexes()]
    created = []
    indexes = {"vector_index_v2": vector_idx, "text_index_v2": text_idx}
    for name, idx in indexes.items():
        if name in existing:
            print(f"  [chunks_v2] search index '{name}' already exists — skipping")
        else:
            chunks_v2.create_search_index(idx)
            created.append(name)
            print(f"  [chunks_v2] search index '{name}' created (may take a few minutes)")
    return created


def setup_standard_indexes_v3(db):
    """Create standard MongoDB indexes for the V3 collections."""
    chunks_v3 = db["chunks_v3"]
    chunks_v3.create_index("crn", name="idx_crn")
    chunks_v3.create_index("course_id", name="idx_course_id")
    chunks_v3.create_index("term", name="idx_term")
    chunks_v3.create_index(
        [("crn", 1), ("chunk_index", 1)],
        unique=True,
        name="idx_crn_chunk_unique",
    )
    print("  [chunks_v3] standard indexes created")

    courses_v3 = db["courses_v3"]
    courses_v3.create_index("crn", unique=True, name="idx_crn")
    courses_v3.create_index("course_id", name="idx_course_id")
    courses_v3.create_index("term", name="idx_term")
    print("  [courses_v3] standard indexes created")


def setup_search_indexes_v3(db):
    """Create Atlas Search and Vector Search indexes for V3 collections."""
    chunks_v3 = db["chunks_v3"]

    vector_idx = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "course_id"},
                {"type": "filter", "path": "term"},
            ]
        },
        name="vector_index_v3",
        type="vectorSearch",
    )

    text_idx = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    "course_id": {"type": "token"},
                    "term": {"type": "token"},
                    "instructor_name": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
        name="text_index_v3",
        type="search",
    )

    existing = [idx["name"] for idx in chunks_v3.list_search_indexes()]
    created = []
    indexes = {"vector_index_v3": vector_idx, "text_index_v3": text_idx}
    for name, idx in indexes.items():
        if name in existing:
            print(f"  [chunks_v3] search index '{name}' already exists — skipping")
        else:
            chunks_v3.create_search_index(idx)
            created.append(name)
            print(f"  [chunks_v3] search index '{name}' created (may take a few minutes)")
    return created


def setup_standard_indexes_v4(db):
    """Create standard MongoDB indexes for the V4 collections."""
    chunks_v4 = db["chunks_v4"]
    chunks_v4.create_index("crn", name="idx_crn")
    chunks_v4.create_index("course_id", name="idx_course_id")
    chunks_v4.create_index("term", name="idx_term")
    chunks_v4.create_index("source", name="idx_source")
    chunks_v4.create_index(
        [("crn", 1), ("chunk_index", 1)],
        unique=True,
        name="idx_crn_chunk_unique",
    )
    print("  [chunks_v4] standard indexes created")

    courses_v4 = db["courses_v4"]
    courses_v4.create_index("crn", unique=True, name="idx_crn")
    courses_v4.create_index("course_id", name="idx_course_id")
    courses_v4.create_index("term", name="idx_term")
    print("  [courses_v4] standard indexes created")


def setup_search_indexes_v4(db):
    """Create Atlas Search and Vector Search indexes for V4 collections."""
    chunks_v4 = db["chunks_v4"]

    vector_idx = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "course_id"},
                {"type": "filter", "path": "term"},
                {"type": "filter", "path": "source"},
            ]
        },
        name="vector_index_v4",
        type="vectorSearch",
    )

    text_idx = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    "header_path": {"type": "string", "analyzer": "lucene.standard"},
                    "course_id": {"type": "token"},
                    "term": {"type": "token"},
                    "source": {"type": "token"},
                    "instructor_name": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
        name="text_index_v4",
        type="search",
    )

    existing = [idx["name"] for idx in chunks_v4.list_search_indexes()]
    created = []
    indexes = {"vector_index_v4": vector_idx, "text_index_v4": text_idx}
    for name, idx in indexes.items():
        if name in existing:
            print(f"  [chunks_v4] search index '{name}' already exists — skipping")
        else:
            chunks_v4.create_search_index(idx)
            created.append(name)
            print(f"  [chunks_v4] search index '{name}' created (may take a few minutes)")
    return created


def setup_indexes_for_collection(db, chunks_col: str) -> list[str]:
    """Create standard + Atlas search indexes for any named chunks collection.

    Derives:
    - courses collection name by replacing the 'chunks_' prefix with 'courses_'
    - index names from the collection name suffix (e.g. chunks_eval → vector_index_eval)

    Adds chunk_tag as a filter field in the vector index and a token field in the
    text index, enabling per-strategy filtered retrieval in eval runs.
    """
    if not chunks_col.startswith("chunks_"):
        raise ValueError(f"Collection name must start with 'chunks_', got: {chunks_col!r}")

    suffix = chunks_col[len("chunks_") :]  # e.g. "eval"
    courses_col = f"courses_{suffix}"
    vector_index_name = f"vector_index_{suffix}"
    text_index_name = f"text_index_{suffix}"

    # Standard indexes
    chunks = db[chunks_col]
    chunks.create_index("crn", name="idx_crn")
    chunks.create_index("course_id", name="idx_course_id")
    chunks.create_index("term", name="idx_term")
    chunks.create_index("chunk_tag", name="idx_chunk_tag")
    chunks.create_index(
        [("crn", 1), ("chunk_index", 1), ("chunk_tag", 1)],
        unique=True,
        name="idx_crn_chunk_tag_unique",
    )
    print(f"  [{chunks_col}] standard indexes created")

    courses = db[courses_col]
    courses.create_index("crn", unique=True, name="idx_crn")
    courses.create_index("course_id", name="idx_course_id")
    courses.create_index("term", name="idx_term")
    print(f"  [{courses_col}] standard indexes created")

    # Atlas Search indexes
    vector_idx = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "course_id"},
                {"type": "filter", "path": "term"},
                {"type": "filter", "path": "chunk_tag"},
            ]
        },
        name=vector_index_name,
        type="vectorSearch",
    )

    text_idx = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {"type": "string", "analyzer": "lucene.standard"},
                    "course_id": {"type": "token"},
                    "term": {"type": "token"},
                    "chunk_tag": {"type": "token"},
                    "instructor_name": {"type": "string", "analyzer": "lucene.standard"},
                },
            }
        },
        name=text_index_name,
        type="search",
    )

    existing = [idx["name"] for idx in chunks.list_search_indexes()]
    created = []
    indexes = {vector_index_name: vector_idx, text_index_name: text_idx}
    for name, idx in indexes.items():
        if name in existing:
            print(f"  [{chunks_col}] search index '{name}' already exists — skipping")
        else:
            chunks.create_search_index(idx)
            created.append(name)
            print(f"  [{chunks_col}] search index '{name}' created (may take a few minutes)")
    return created


def main():
    parser = argparse.ArgumentParser(description="Create MongoDB Atlas indexes")
    parser.add_argument(
        "--version",
        choices=["v1", "v2", "v3", "v4", "all"],
        default="all",
        help="Which index set to create (default: all)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Create indexes for a specific chunks collection (e.g. chunks_eval). Overrides --version.",
    )
    args = parser.parse_args()

    db = get_db()
    print(f"Setting up indexes on database '{DB_NAME}'...\n")

    if args.collection:
        print(f"Custom collection — {args.collection}:")
        search_indexes = setup_indexes_for_collection(db, args.collection)
    else:
        if args.version in ("v1", "all"):
            print("V1 — Standard indexes:")
            setup_standard_indexes(db)
            print("\nV1 — Atlas Search indexes:")
            search_indexes = setup_search_indexes(db)
        else:
            search_indexes = []

        if args.version in ("v2", "all"):
            print("\nV2 — Standard indexes:")
            setup_standard_indexes_v2(db)
            print("\nV2 — Atlas Search indexes:")
            search_indexes += setup_search_indexes_v2(db)

        if args.version in ("v3", "all"):
            print("\nV3 — Standard indexes:")
            setup_standard_indexes_v3(db)
            print("\nV3 — Atlas Search indexes:")
            search_indexes += setup_search_indexes_v3(db)

        if args.version in ("v4", "all"):
            print("\nV4 — Standard indexes:")
            setup_standard_indexes_v4(db)
            print("\nV4 — Atlas Search indexes:")
            search_indexes += setup_search_indexes_v4(db)

    print("\nDone.")
    if search_indexes:
        col = args.collection or "chunks_v4"
        print(
            "NOTE: Atlas Search indexes build asynchronously. "
            f"Check Atlas UI or run db.{col}.getSearchIndexes() to verify status."
        )


if __name__ == "__main__":
    main()
