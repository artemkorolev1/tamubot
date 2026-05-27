.PHONY: run scrape-catalog scrape-classes scrape-simple-syllabus setup-atlas ingest ingest-dept \
        ingest-corpus test typecheck lint format probe probe-v3 probe-full \
        eval-draft import-draft bench bench-ragas test-v4 probe-v4 \
        eval eval-gen eval-chunking diff-runs ragas-testset \
        sandbox-up sandbox-down sandbox-shell agent \
        dagster-v6b dagster-v6c

# --- App ---
run:
	@echo "Starting TamuBot..."
	@streamlit run src/tamubot/app/streamlit.py --server.headless true

# --- Data Pipeline ---
scrape-catalog:
	scrapy crawl catalog

scrape-classes:
	scrapy crawl class_search

scrape-simple-syllabus:
	python -m tamubot.scraper.download_simple_syllabus

setup-atlas:
	python -m tamubot.ingestion.setup_atlas

ingest:
	python -m tamubot.ingestion.ingest

ingest-v3:
	python -m tamubot.ingestion.ingest --v3

ingest-dept:
	python -m tamubot.ingestion.ingest --department $(DEPT)

ingest-corpus:
	python -m tamubot.ingestion.ingest --v3 --crns-file tamu_data/evals/eval_corpus.json

# --- Dev / Testing ---
test:
	pytest tests/ -v

typecheck:
	mypy src/tamubot/ --ignore-missing-imports

lint:
	ruff check src/tamubot/

format:
	ruff format src/tamubot/

probe:
	python -m tamubot.evals.run_probe --suite smoke

probe-v3:
	USE_V4_PIPELINE=false python -m tamubot.evals.run_probe --suite smoke

probe-full:
	python -m tamubot.evals.run_probe --suite all

test-v4:
	pytest tests/test_v4_*.py -v

probe-v4:
	python -m tamubot.evals.run_probe --suite smoke

# --- Benchmarking ---
ragas-testset:  ## Generate RAGAS testset from eval corpus
	python -m tamubot.evals.generate_ragas_testset --corpus-dir $(CORPUS_DIR) $(ARGS)

eval-draft:
	python -m tamubot.evals.generate_eval_draft --n 60

import-draft:
	python -m tamubot.evals.import_eval_draft --draft $(DRAFT) --tag $(or $(TAG),v1)

bench:
	python -m tamubot.evals.run_benchmark --golden-set $(GOLDEN) --experiment-name $(EXP) \
		$(if $(CHUNKS_COL),--chunks-collection $(CHUNKS_COL),)

bench-ragas:
	python -m tamubot.evals.run_benchmark --golden-set $(GOLDEN) --experiment-name $(EXP) --ragas \
		$(if $(CHUNKS_COL),--chunks-collection $(CHUNKS_COL),)

eval-chunking:
	SESSION_CACHE_ENABLED=false python -m tamubot.evals.eval_chunking \
		--golden-set $(GOLDEN) \
		--experiment $(EXP) \
		$(if $(RAGAS),--ragas,) \
		$(if $(TOP_K),--top-k $(TOP_K),) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD),) \
		$(if $(CHUNKS_COL),--chunks-collection $(CHUNKS_COL),) \
		$(if $(DESC),--description "$(DESC)",)

# --- Unified eval runner (recommended) -------------------------------------
# Required: GOLDEN, EXP
# Optional: METRICS=faithfulness,context_recall  IDS=3,7,12  CAPTURE=1
#           TOP_K=  THRESHOLD=  CHUNKS_COL=  CHUNK_TAG=  DESC="..."
eval:
	SESSION_CACHE_ENABLED=false python -m tamubot.evals.run_eval \
		--golden-set $(GOLDEN) \
		--experiment $(EXP) \
		$(if $(METRICS),--metrics $(METRICS),) \
		$(if $(IDS),--ids $(IDS),) \
		$(if $(CAPTURE),--capture-state,) \
		$(if $(TOP_K),--top-k $(TOP_K),) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD),) \
		$(if $(CHUNKS_COL),--chunks-collection $(CHUNKS_COL),) \
		$(if $(CHUNK_TAG),--chunk-tag $(CHUNK_TAG),) \
		$(if $(DESC),--description "$(DESC)",)

eval-gen:
	SESSION_CACHE_ENABLED=false python -m tamubot.evals.run_eval \
		--golden-set $(GOLDEN) \
		--experiment $(EXP) \
		--with-generation \
		$(if $(METRICS),--metrics $(METRICS),) \
		$(if $(IDS),--ids $(IDS),) \
		$(if $(CAPTURE),--capture-state,) \
		$(if $(TOP_K),--top-k $(TOP_K),) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD),) \
		$(if $(CHUNKS_COL),--chunks-collection $(CHUNKS_COL),) \
		$(if $(CHUNK_TAG),--chunk-tag $(CHUNK_TAG),) \
		$(if $(DESC),--description "$(DESC)",)

# Compare two run:<exp> columns from a golden set.
# Required: GOLDEN, LEFT (e.g. run:foo), RIGHT (e.g. run:bar), OUTPUT
diff-runs:
	python -m tamubot.evals.diff_runs \
		--golden-set $(GOLDEN) \
		--left $(LEFT) \
		--right $(RIGHT) \
		--output $(OUTPUT) \
		$(if $(METRIC),--metric $(METRIC),)

# --- Docker Sandbox ---
sandbox-up:
	docker compose up -d

sandbox-down:
	docker compose down

sandbox-shell:
	docker exec -it tamubot-dev-1 bash

agent:
	docker exec -it tamubot-dev-1 claude --dangerously-skip-permissions

# --- Dagster UI ---
dagster-v6b:   ## launch Dagster UI for pipeline_v6b at localhost:3000
	dagster dev -f src/tamubot/ingestion/pipeline_v6b/definitions.py

dagster-v6c:   ## launch Dagster UI for pipeline_v6c at localhost:3001
	dagster dev -f src/tamubot/ingestion/pipeline_v6c/definitions.py --port 3001
