"""Postgres (pgvector) retrieval backend — Phase 1 of the MongoDB → Postgres migration.

Holds the schema DDL (`schema.sql`), the idempotent setup entry point
(`setup_postgres`), and the connection pool helper (`pool`).
"""
