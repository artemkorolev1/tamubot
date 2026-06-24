-- TamuBot Postgres schema — Phase 1 (Option 2: unified corpus + promoted structured tables).
--
-- Design notes:
--   * One `chunks` table with a `doc_type` discriminator carries EVERY document
--     type (syllabus, catalog_course, program, policy, calendar) through a single
--     HNSW vector index and a single retrieval path. New doc types are inserts,
--     not migrations. `doc_type` is CHECK-constrained so a typo can't ghost rows.
--   * The relational course core (courses / sections / instructors) makes
--     structured filters — instructor, term, topic, grading — plain SQL joins.
--   * The structured tables promote the previously-dormant SyllabusExtract
--     (silver/05_structured/*.json) and the course_summary "Topics:" line into
--     queryable rows. This is what unlocks the agentic query API.
--   * Expansion tables (programs / program_requirements / policies /
--     calendar_events) are created now but populated later (Phase 3+).
--
-- Table order respects FK dependencies (instructors/courses → sections →
-- documents → chunks → section children). The whole script is idempotent
-- (IF NOT EXISTS everywhere; constraints guarded in DO blocks). Apply via
-- `python -m tamubot.ingestion.postgres.setup_postgres`.

CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector >= 0.8 (iterative scans for filtered ANN)
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- trigram fuzzy match (instructor / title / topic search)

-- ─────────────────────────────────────────────────────────────────────────────
-- Relational course core  (created first; documents/chunks reference it)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS instructors (
    id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name  text NOT NULL,
    email text,
    dept  text,
    UNIQUE (name, dept)
);
-- Fuzzy name lookup for find_instructor("smith").
CREATE INDEX IF NOT EXISTS idx_instructors_name_trgm ON instructors USING gin (name gin_trgm_ops);

CREATE TABLE IF NOT EXISTS courses (              -- term-independent (catalog level)
    course_id           text PRIMARY KEY,         -- 'CSCE 624'
    dept                text,
    number              text,
    title               text,
    catalog_description text,                      -- expansion: from catalog scraper
    credit_hours        text,                      -- free text in the source ("3", "3-4", "Variable")
    prerequisites_text  text
);
CREATE INDEX IF NOT EXISTS idx_courses_dept       ON courses (dept);
CREATE INDEX IF NOT EXISTS idx_courses_title_trgm ON courses USING gin (title gin_trgm_ops);

CREATE TABLE IF NOT EXISTS sections (             -- one row per CRN
    crn                       text PRIMARY KEY,
    course_id                 text REFERENCES courses(course_id) ON UPDATE CASCADE,
    section                   text,
    term                      text,
    instructor_id             bigint REFERENCES instructors(id),
    meeting_times             text,                -- raw string from courses_v4 (kept verbatim)
    location                  text,
    format                    text,                -- 'online'|'in-person'
    credit_hours              text,                -- free text in the source
    course_type               text,                -- 'seminar'|'research'|... (from classify_course_type)
    syllabus_url              text,
    source                    text,                -- 'howdy_portal'|'simple_syllabus'
    course_summary            text,
    -- promoted SyllabusExtract policy paragraphs (1:1 with the section)
    attendance_policy         text,
    academic_integrity_policy text,
    ingested_at               timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sections_course     ON sections (course_id);
CREATE INDEX IF NOT EXISTS idx_sections_term       ON sections (term);
CREATE INDEX IF NOT EXISTS idx_sections_instructor ON sections (instructor_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- Unified document + chunk core  (all doc_types flow through here)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_type    text NOT NULL,            -- 'syllabus'|'catalog_course'|'program'|'policy'|'calendar'
    title       text,
    source      text,                     -- 'howdy_portal'|'simple_syllabus'|'catalog'|...
    source_url  text,
    source_file text,                     -- the stem; natural key from the pipeline
    crn         text REFERENCES sections(crn) ON UPDATE CASCADE,  -- N:1 doc→section (a CRN may have 2 source docs: HP + Simple Syllabus)
    dept        text,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    sha256      text,
    UNIQUE (source_file, doc_type)
);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents (doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_dept     ON documents (dept);
CREATE INDEX IF NOT EXISTS idx_documents_crn      ON documents (crn);   -- FK support + section→documents lookup

CREATE TABLE IF NOT EXISTS chunks (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id     bigint REFERENCES documents(id) ON DELETE CASCADE,
    doc_type        text NOT NULL DEFAULT 'syllabus',  -- denormalized for fast filtering
    source_file     text NOT NULL,
    chunk_index     int  NOT NULL,
    chunk_tag       text NOT NULL DEFAULT 'semantic',
    -- denormalized course context (null for non-syllabus chunks; source of truth is `sections`)
    course_id       text,
    crn             text,
    term            text,
    section         text,
    instructor_name text,
    source          text,              -- 'howdy_portal'|'simple_syllabus' (origin; mirrors chunks_v4.source)
    -- content
    content         text NOT NULL,
    header_path     text,
    anchor          text,
    page            int,
    token_count     int,
    has_table       boolean NOT NULL DEFAULT false,
    split_reason    text,
    flags           text[],
    -- v6 / v6b tagging
    is_boilerplate      boolean NOT NULL DEFAULT false,
    boilerplate_cluster text,
    cluster_confidence  real,
    is_duplicate        boolean NOT NULL DEFAULT false,
    -- vector + lexical
    embedding       vector(1024),         -- Voyage voyage-3 (1024-d). halfvec(1024) halves storage if needed.
    embedding_model text,
    -- header_path weighted above body so a header lexical hit ranks higher (ts_rank_cd uses weights).
    content_tsv     tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', coalesce(header_path, '')), 'A') ||
                        setweight(to_tsvector('english', content), 'B')
                    ) STORED,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_file, chunk_index, chunk_tag)   -- the pipeline upsert grain (source_file, not crn: HP+SS copies of one CRN must not collide)
);

-- One ANN index across every doc_type (cosine, matching the Atlas setting).
-- Partial on embedding IS NOT NULL: un-embedded chunks (pre-embed stage) are
-- excluded from the index and from vector search (queries filter the same way).
-- The hot path must enable `hnsw.iterative_scan` + raise `hnsw.ef_search` so
-- course_id/doc_type filtering doesn't under-return (see queries.py).
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;
-- BM25-flavored full-text (weighted header_path + content).
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (content_tsv);
-- Relational filter index — leads with the equality columns actually filtered
-- together on the hot path (chunk_tag defaults to 'semantic'; boilerplate/dup
-- are handled by the partial predicate, not as index keys).
CREATE INDEX IF NOT EXISTS idx_chunks_filters ON chunks (doc_type, chunk_tag, course_id, term);
-- Ordered per-course scan for fetch_anchor_chunks (WHERE course_id ORDER BY chunk_index).
CREATE INDEX IF NOT EXISTS idx_chunks_course_idx ON chunks (course_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
-- Instructor-scoped chunk search (fuzzy, denormalized name).
CREATE INDEX IF NOT EXISTS idx_chunks_instructor_trgm ON chunks USING gin (instructor_name gin_trgm_ops);

-- ─────────────────────────────────────────────────────────────────────────────
-- Promoted SyllabusExtract  (was dormant in silver/05_structured/*.json)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS section_assessments (   -- SyllabusExtract.assessment_weights[]
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    crn         text REFERENCES sections(crn) ON DELETE CASCADE ON UPDATE CASCADE,
    component   text,
    weight_pct  numeric(5, 2) CHECK (weight_pct IS NULL OR (weight_pct >= 0 AND weight_pct <= 100))
);
CREATE INDEX IF NOT EXISTS idx_section_assessments_crn       ON section_assessments (crn);
CREATE INDEX IF NOT EXISTS idx_section_assessments_component ON section_assessments (lower(component));

CREATE TABLE IF NOT EXISTS section_grade_cutoffs (  -- SyllabusExtract.letter_grade_cutoffs[]
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    crn         text REFERENCES sections(crn) ON DELETE CASCADE ON UPDATE CASCADE,
    grade       text,
    min_percent numeric(5, 2) CHECK (min_percent IS NULL OR (min_percent >= 0 AND min_percent <= 100))
);
CREATE INDEX IF NOT EXISTS idx_section_grade_cutoffs_crn   ON section_grade_cutoffs (crn);
CREATE INDEX IF NOT EXISTS idx_section_grade_cutoffs_grade ON section_grade_cutoffs (grade);

CREATE TABLE IF NOT EXISTS learning_outcomes (      -- SyllabusExtract.learning_outcomes[]
    id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    crn     text REFERENCES sections(crn) ON DELETE CASCADE ON UPDATE CASCADE,
    ordinal int,
    text    text
);
CREATE INDEX IF NOT EXISTS idx_learning_outcomes_crn ON learning_outcomes (crn);

CREATE TABLE IF NOT EXISTS section_meetings (       -- SyllabusExtract.meeting_schedule[]
    id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    crn      text REFERENCES sections(crn) ON DELETE CASCADE ON UPDATE CASCADE,
    ordinal  int,
    day      text,
    time     text,
    location text
);
CREATE INDEX IF NOT EXISTS idx_section_meetings_crn ON section_meetings (crn);

CREATE TABLE IF NOT EXISTS section_prerequisites (  -- SyllabusExtract.prerequisites[]
    id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    crn     text REFERENCES sections(crn) ON DELETE CASCADE ON UPDATE CASCADE,
    ordinal int,
    text    text
);
CREATE INDEX IF NOT EXISTS idx_section_prerequisites_crn ON section_prerequisites (crn);

-- Topics come from the per-section course_summary "Topics:" line (already
-- generated today). Stored at SECTION grain (its true source); the
-- course_topics view aggregates to course level for list_courses(topic=...).
CREATE TABLE IF NOT EXISTS section_topics (
    id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    crn     text REFERENCES sections(crn) ON DELETE CASCADE ON UPDATE CASCADE,
    topic   text NOT NULL,
    UNIQUE (crn, topic)
);
CREATE INDEX IF NOT EXISTS idx_section_topics_topic      ON section_topics (lower(topic));
CREATE INDEX IF NOT EXISTS idx_section_topics_topic_trgm ON section_topics USING gin (topic gin_trgm_ops);

-- Course-level topic discovery (term-independent union over the course's sections).
CREATE OR REPLACE VIEW course_topics AS
    SELECT DISTINCT s.course_id, t.topic
    FROM section_topics t
    JOIN sections s ON s.crn = t.crn
    WHERE s.course_id IS NOT NULL;

-- ─────────────────────────────────────────────────────────────────────────────
-- Expansion (created now, populated in a later phase — see Option 3)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS programs (
    id                      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name                    text NOT NULL,
    level                   text,             -- 'masters'|'phd'|'certificate'
    dept                    text,
    description_document_id bigint REFERENCES documents(id)
);
CREATE INDEX IF NOT EXISTS idx_programs_description_doc ON programs (description_document_id);

CREATE TABLE IF NOT EXISTS program_requirements (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    program_id         bigint REFERENCES programs(id) ON DELETE CASCADE,
    requirement_text   text,
    required_course_id text REFERENCES courses(course_id) ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_program_requirements_program ON program_requirements (program_id);
CREATE INDEX IF NOT EXISTS idx_program_requirements_course  ON program_requirements (required_course_id);

CREATE TABLE IF NOT EXISTS policies (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text NOT NULL,
    scope       text,                      -- 'university'|'college'|'dept'
    full_text   text,
    document_id bigint REFERENCES documents(id)
);
CREATE INDEX IF NOT EXISTS idx_policies_document ON policies (document_id);

CREATE TABLE IF NOT EXISTS calendar_events (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    term       text,
    name       text,
    event_date date,
    kind       text                        -- 'registration'|'drop'|'finals'|...
);
CREATE INDEX IF NOT EXISTS idx_calendar_events_term ON calendar_events (term);

-- ─────────────────────────────────────────────────────────────────────────────
-- Constraints that lack an IF NOT EXISTS form — guarded for idempotency.
-- doc_type is the discriminator the whole design pivots on; constrain it so a
-- typo ('sylabus') can't silently partition the corpus into a ghost type.
-- ─────────────────────────────────────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'documents_doc_type_chk') THEN
        ALTER TABLE documents ADD CONSTRAINT documents_doc_type_chk
            CHECK (doc_type IN ('syllabus', 'catalog_course', 'program', 'policy', 'calendar'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chunks_doc_type_chk') THEN
        ALTER TABLE chunks ADD CONSTRAINT chunks_doc_type_chk
            CHECK (doc_type IN ('syllabus', 'catalog_course', 'program', 'policy', 'calendar'));
    END IF;
END $$;
