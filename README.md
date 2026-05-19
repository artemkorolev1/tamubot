# TamuBot

A RAG-based academic assistant for Texas A&M University students. Ask questions about courses, syllabi, grading policies, schedules, and university policies — get cited answers grounded in real syllabus data.

## Architecture

TamuBot follows a **3-stage RAG pipeline** orchestrated by LangGraph:

- **Router** — Gemini 2.5 Flash extracts structured variables (course IDs, categories, intent type) from the user query. A pure-Python function matrix derives the retrieval strategy — no ML classification step.
- **Retrieval** — Three search paths depending on query type: metadata lookup (exact index), hybrid search (RRF: vector + BM25), or semantic search (full-corpus vector). Results are reranked by Voyage AI cross-encoder.
- **Generator** — Gemini 2.5 Flash produces a cited answer using function-adaptive prompts and XML-formatted context. Every claim links back to a `[Source N]` citation.

The pipeline runs as a LangGraph state machine with conversation memory (mem0 Cloud), session caching, and SQLite checkpointing.

![Architecture](docs/architecture.png)

## Tech Stack

### Frontend

- **Streamlit** — Chat UI with session state management

### Backend / RAG Pipeline

- **LangGraph** — State machine orchestration with conditional routing
- **Gemini 2.5 Flash** — Router, generator, and PDF parsing
- **Gemini 2.5 Flash Lite** — Validation model
- **MongoDB Atlas** — Vector store, full-text search, and metadata indexes
- **Voyage AI** — `voyage-3` embeddings (1024-dim) and `rerank-2` cross-encoder
- **mem0 Cloud** — Conversational memory across sessions
- **Pydantic v2** — Schema validation for all data models

### Data Pipeline

- **Scrapy** — Course catalog and class schedule spiders
- **Playwright** — Simple Syllabus PDF downloader (bypasses CloudFront WAF)
- **Gemini 2.5 Flash** — Multimodal PDF → structured JSON parsing (13 categories)

### Observability

- **Langfuse** — End-to-end request tracing (Router → Retrieval → Generator)
- **RAGAS** — Async background evaluation (Faithfulness + Answer Relevancy)
- **OpenTelemetry** — Instrumentation layer

### Infrastructure

- **Docker** — Containerized development environment (Python 3.14-slim)
- **Docker Compose** — App + API rate-limiting proxy
- **API Proxy** — Per-session rate budgets for TAMU API and Voyage AI

### Development

- **pytest** — Unit and integration tests
- **mypy** — Static type checking
- **ruff** — Linting and formatting

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

| Variable | Required | Description |
|---|---|---|
| `MONGODB_URI` | Yes | MongoDB Atlas connection string |
| `MONGODB_DB` | No | Database name (default: `tamubot`) |
| `VOYAGE_API_KEY` | Yes | Voyage AI API key |
| `GOOGLE_API_KEY` | Yes | Google AI API key (Gemini) |
| `TAMU_API_KEY` | No | TAMU AI gateway key (routes all RAG LLM calls when set) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |
| `LANGFUSE_BASE_URL` | No | Langfuse host (default: `https://cloud.langfuse.com`) |

See `.env.example` for the full list including rate limits, proxy config, and legacy Vertex AI settings.

## Features

- **Intelligent Query Routing** — 8-function derivation matrix handles metadata lookups, hybrid search, semantic discovery, multi-course comparisons, and out-of-scope rejection
- **Multi-Course Comparison** — Structured side-by-side tables with per-cell citations
- **Recursive Course Discovery** — 5-step pipeline finds related courses anchored to a named course (e.g., "What should I take alongside CSCE 638?")
- **Intent-Aware Generation** — Advisory overlays for academic, career, difficulty, and planning queries
- **Data Integrity Flags** — Disclaimers when syllabus data is missing for requested courses/categories
- **Conversational Memory** — mem0 Cloud preserves context across sessions
- **Full Observability** — Every query traced end-to-end in Langfuse with automated RAGAS scoring
- **Citation System** — All answers include `[Source N]` references to specific syllabus chunks

## Deployment

### Docker (recommended)

```
docker compose build
```

```
docker compose up
```

Open [http://localhost:8501](http://localhost:8501).

### Local Development

```bash
git clone https://github.com/artemkorolev1/tamubot
cd tamubot
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env

# One-time: create MongoDB Atlas indexes
python -m tamubot.ingestion.setup_atlas

# Ingest parsed syllabi into MongoDB
python -m tamubot.ingestion.ingest

# Start the app
streamlit run src/tamubot/app/streamlit.py --server.headless true
```

## Project Structure

```
tamubot/
├── Makefile                       # Dev targets (run, ingest, probe, bench, ...)
├── Dockerfile                     # Python 3.14-slim container
├── docker-compose.yml             # App + API proxy
├── pyproject.toml                 # Project metadata + tool config
├── requirements.txt               # Pinned runtime dependencies
│
├── src/tamubot/                   # Main package
│   ├── app/                       # Streamlit chat UI
│   │   ├── streamlit.py           # Entry point
│   │   └── citations.py           # [Source N] rendering
│   ├── core/                      # Shared config + env loading
│   │   └── config.py
│   ├── rag/                       # Query-time RAG pipeline
│   │   ├── router.py              # Variable extraction + function derivation
│   │   ├── generator.py           # Function-adaptive prompts + citations
│   │   ├── prompts.py             # System prompt templates
│   │   ├── graph/                 # LangGraph state machine
│   │   ├── nodes/                 # Router, retrieval, generator nodes
│   │   ├── edges/                 # Conditional routing logic
│   │   ├── tools/                 # MongoDB, Voyage, mem0, LLM clients
│   │   ├── state/                 # LangGraph state definitions
│   │   └── observability/         # Langfuse tracing + RAGAS evaluation
│   ├── ingestion/                 # ETL: scrape → parse → embed → store
│   │   ├── process_syllabi.py     # Gemini PDF → structured JSON
│   │   ├── ingest.py              # Validate + embed + MongoDB upsert
│   │   ├── setup_atlas.py         # Create Atlas indexes
│   │   ├── chunker_v4.py          # Token-aware chunker
│   │   ├── converters/            # Docling, Gemini PDF backends
│   │   ├── validators/            # Schema + content checks
│   │   └── filters/               # Boilerplate stripping
│   ├── evals/                     # Evaluation suite
│   │   ├── run_probe.py           # Smoke + full end-to-end probes
│   │   ├── run_benchmark.py       # Golden-set benchmarking
│   │   ├── eval_chunking.py       # Chunking strategy comparison
│   │   └── golden_set.py          # Golden set management
│   ├── scraper/                   # Scrapy spiders + Playwright downloaders
│   └── advisory/                  # Intent-aware advisory overlays
│
├── tamu_data/                     # Scraped data, parsed JSONs, logs (gitignored payloads)
│   ├── processed/                 # Structured JSON outputs
│   ├── raw/                       # PDFs + JSONL
│   └── evals/                     # Golden sets + reports
│
├── tools/api-proxy/               # Rate-limiting reverse proxy
├── scripts/                       # One-off maintenance + reporting scripts
├── tests/                         # pytest suite
└── docs/                          # Architecture diagram + public docs
```

## Data Pipeline

Scrape → parse → embed → store. Run once to populate MongoDB, or re-run to refresh data.

```bash
# 1. Scrape course catalog + class sections
make scrape-catalog
make scrape-classes

# 2. Download graduate syllabi (Playwright)
make scrape-simple-syllabus

# 3. Parse PDFs with Gemini
GOOGLE_API_KEY=... python -m tamubot.ingestion.process_syllabi

# 4. Create Atlas indexes + ingest
python -m tamubot.ingestion.setup_atlas
python -m tamubot.ingestion.ingest

# Single department only
python -m tamubot.ingestion.ingest --department CSCE

# Preview without writing to DB
python -m tamubot.ingestion.ingest --dry-run
```

## Evaluation

```bash
# Run pytest suite
make test

# End-to-end smoke test
make probe

# Full probe suite
make probe-full

# Golden-set benchmark
make bench GOLDEN=path/to/golden.xlsx EXP=experiment-name

# Chunking strategy evaluation
make eval-chunking GOLDEN=path/to/golden.xlsx EXP=experiment-name
```

## License

MIT
