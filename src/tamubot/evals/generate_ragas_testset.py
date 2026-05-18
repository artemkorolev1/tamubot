"""RAGAS Testset Generator for TamuBot evaluation.

Generates document-grounded QA pairs from syllabus files using RAGAS's
TestsetGenerator and Knowledge Graph. Outputs an XLSX golden set compatible
with run_benchmark.py (same SCHEMA_COLUMNS).

Usage:
    python -m tamubot.evals.generate_ragas_testset --corpus-dir <path>
    python -m tamubot.evals.generate_ragas_testset --corpus-dir <path> --dry-run
    python -m tamubot.evals.generate_ragas_testset --corpus-dir <path> --testset-size 10
    python -m tamubot.evals.generate_ragas_testset --corpus-dir <path> --provider tamu
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from tamubot.core import config

logger = logging.getLogger("tamubot.evals.ragas_testset")

# ---------------------------------------------------------------------------
# Synthesizer name → expected_function mapping
# ---------------------------------------------------------------------------

SYNTHESIZER_TO_FUNCTION: dict[str, str] = {
    "single_hop_specific_query_synthesizer": "hybrid_course",
    "multi_hop_specific_query_synthesizer": "recursive",
    "multi_hop_abstract_query_synthesizer": "semantic_general",
}

DISTRIBUTION_PRESETS = ("default", "balanced_50_50", "semantic_only")

# ---------------------------------------------------------------------------
# Step 1 — Document loading
# ---------------------------------------------------------------------------

# Filename pattern: {TERM}_{DEPT}_{COURSENUM}_{SECTION}_{CRN}_v{VERSION}.json
_FILENAME_RE = re.compile(
    r"^(?P<term>\d+)_(?P<dept>[A-Z]+)_(?P<coursenum>\d+)_(?P<section>\d+)_(?P<crn>\d+)_v\d+\.json$"
)


def load_corpus(corpus_dir: Path, include_terms: set[str] | None = None) -> list[Document]:
    """Load v3_step3_flat / v4 06_chunk JSON files as LangChain Documents.

    Each document gets the full concatenated Markdown as page_content.
    RAGAS handles its own internal segmentation during KG construction.

    If `include_terms` is provided, only files whose filename starts with one
    of those term codes (e.g. {"202611", "202641"}) are loaded.
    """
    json_files = sorted(corpus_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {corpus_dir}")

    docs: list[Document] = []
    for fp in json_files:
        if fp.name.startswith("_"):
            continue  # skip metadata files like _run_meta_*.json
        if include_terms is not None:
            term_prefix = fp.name.split("_", 1)[0]
            if term_prefix not in include_terms:
                continue

        with open(fp) as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        if not chunks:
            logger.warning("Skipping %s — no chunks", fp.name)
            continue

        page_content = "\n\n".join(c["content"] for c in chunks)

        meta = data.get("course_metadata", {})
        crn = meta.get("crn")
        course_id = meta.get("course_id", "")
        term = meta.get("term", "")
        section = meta.get("section", "")

        # Fallback: parse CRN from filename if metadata has null CRN
        if not crn:
            m = _FILENAME_RE.match(fp.name)
            if m:
                crn = m.group("crn")
                if not course_id:
                    course_id = f"{m.group('dept')} {m.group('coursenum')}"

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "crn": crn or "",
                    "course_id": course_id,
                    "term": term,
                    "section": section,
                    "source_file": fp.name,
                },
            )
        )

    logger.info("Loaded %d documents from %s", len(docs), corpus_dir)
    return docs


# ---------------------------------------------------------------------------
# Step 2 — LLM setup
# ---------------------------------------------------------------------------


def build_llm(provider: str, temperature: float):
    """Create a RAGAS-compatible LLM via llm_factory.

    Ragas 0.4.x requires a client instance — text-only mode was removed.
    Google uses the Instructor adapter wrapping google.genai.Client.
    """
    from ragas.llms import llm_factory

    if provider == "google":
        from google import genai

        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        return llm_factory(
            "gemini-2.0-flash",
            provider="google",
            client=client,
            temperature=temperature,
        )

    if provider == "tamu":
        from openai import OpenAI

        from tamubot.evals.tamu_openai_workaround import wrap_for_tamu

        client = wrap_for_tamu(
            OpenAI(
                api_key=config.TAMU_API_KEY,
                base_url=config.TAMU_BASE_URL,
            )
        )
        return llm_factory(
            config.TAMU_MODEL,
            provider="openai",
            client=client,
            temperature=temperature,
        )

    raise ValueError(f"Unknown provider: {provider!r}. Use 'google' or 'tamu'.")


# ---------------------------------------------------------------------------
# Step 3 — Embedding setup
# ---------------------------------------------------------------------------


def build_embeddings():
    """Google gemini-embedding-001 wrapped for RAGAS.

    Note: text-embedding-004 was removed from Google's v1beta endpoint
    in 2026. gemini-embedding-001 is the current replacement (3072-dim).
    TAMU's gateway has no embedding endpoint, so embeddings always go
    to Google direct regardless of --provider.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=config.GOOGLE_API_KEY,
        )
    )


# ---------------------------------------------------------------------------
# Step 4 — Knowledge Graph construction
# ---------------------------------------------------------------------------


def _build_syllabus_ner_prompt():
    """NER prompt tuned for syllabus entities, ignoring shared boilerplate."""
    from ragas.testset.transforms.extractors.llm_based import (
        NEROutput,
        NERPrompt,
        TextWithExtractionLimit,
    )

    class SyllabusNERPrompt(NERPrompt):
        instruction: str = (
            "Extract named entities from the given syllabus text. "
            "Focus on: course IDs (e.g. CSCE 638), instructor names, "
            "grading components and weights, prerequisites, textbooks, "
            "exam formats, project types, tools/software, credit hours, "
            "and specific deadlines or dates.\n\n"
            "IGNORE the following boilerplate that is identical across syllabi: "
            "honor code statements, ADA/disability policy, university attendance policy, "
            "FERPA notice, mental health resources, Title IX statements, "
            "and academic integrity definitions. "
            "Do not extract entities from these sections.\n\n"
            "Limit output to the top entities. "
            "Ensure the number of entities does not exceed the specified maximum."
        )
        examples: list[tuple[TextWithExtractionLimit, NEROutput]] = [
            (
                TextWithExtractionLimit(
                    text=(
                        "CSCE 638 - Section 600 - Algorithms\n"
                        "Instructor: Dr. Fang Song\n"
                        "Prerequisites: CSCE 411\n"
                        "Grading: Homework 40%, Midterm 25%, Final 35%\n"
                        "Textbook: Introduction to Algorithms by Cormen et al.\n"
                        "An Aggie does not lie, cheat, or steal."
                    ),
                    max_num=10,
                ),
                NEROutput(
                    entities=[
                        "CSCE 638",
                        "Dr. Fang Song",
                        "CSCE 411",
                        "Homework 40%",
                        "Midterm 25%",
                        "Final 35%",
                        "Introduction to Algorithms",
                        "Cormen et al.",
                    ]
                ),
            ),
        ]

    return SyllabusNERPrompt()


def build_transforms(llm, embedding_model, documents: list[Document]):
    """Custom KG transforms adapted from RAGAS default_transforms.

    Uses a syllabus-specific NER prompt to avoid boilerplate cross-links.
    """
    from ragas.testset.graph import NodeType
    from ragas.testset.transforms import Parallel
    from ragas.testset.transforms.extractors import (
        EmbeddingExtractor,
        HeadlinesExtractor,
        NERExtractor,
        SummaryExtractor,
    )
    from ragas.testset.transforms.extractors.llm_based import ThemesExtractor
    from ragas.testset.transforms.filters import CustomNodeFilter
    from ragas.testset.transforms.relationship_builders import (
        CosineSimilarityBuilder,
        OverlapScoreBuilder,
    )
    from ragas.testset.transforms.splitters import HeadlineSplitter
    from ragas.utils import num_tokens_from_string

    def filter_long_docs(node):
        return node.type == NodeType.DOCUMENT and num_tokens_from_string(node.properties.get("page_content", "")) > 500

    def filter_docs(node):
        return node.type == NodeType.DOCUMENT

    def filter_chunks(node):
        return node.type == NodeType.CHUNK

    # Check document length distribution to decide transform pipeline
    token_counts = [num_tokens_from_string(doc.page_content) for doc in documents]
    long_pct = sum(1 for t in token_counts if t > 500) / len(token_counts)

    if long_pct < 0.25:
        raise ValueError(
            f"Only {long_pct:.0%} of documents exceed 500 tokens. "
            "Syllabus documents should be longer — check your corpus."
        )

    headline_extractor = HeadlinesExtractor(llm=llm, filter_nodes=lambda node: filter_long_docs(node))
    # Splitter must skip short docs that HeadlinesExtractor never set
    # 'headlines' on — otherwise it raises ValueError and crashes the run.
    splitter = HeadlineSplitter(min_tokens=500, filter_nodes=lambda node: filter_long_docs(node))
    summary_extractor = SummaryExtractor(llm=llm, filter_nodes=lambda node: filter_long_docs(node))
    node_filter = CustomNodeFilter(llm=llm, filter_nodes=lambda node: filter_chunks(node))

    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=lambda node: filter_long_docs(node),
    )
    theme_extractor = ThemesExtractor(llm=llm, filter_nodes=lambda node: filter_chunks(node))
    ner_extractor = NERExtractor(
        llm=llm,
        filter_nodes=lambda node: filter_chunks(node),
        prompt=_build_syllabus_ner_prompt(),
    )

    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.7,
        filter_nodes=lambda node: filter_long_docs(node),
    )
    ner_overlap_sim = OverlapScoreBuilder(threshold=0.01, filter_nodes=lambda node: filter_chunks(node))

    return [
        headline_extractor,
        splitter,
        summary_extractor,
        node_filter,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]


def build_or_load_kg(
    documents: list[Document],
    llm,
    embedding_model,
    kg_path: Path,
    rebuild: bool,
):
    """Load cached KG or build from scratch."""
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms

    if kg_path.exists() and not rebuild:
        logger.info("Loading cached KG from %s", kg_path)
        return KnowledgeGraph.load(str(kg_path))

    logger.info("Building Knowledge Graph from %d documents...", len(documents))
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )
    logger.info("Seeded KG with %d DOCUMENT nodes", len(kg.nodes))
    transforms = build_transforms(llm, embedding_model, documents)
    apply_transforms(kg, transforms)
    kg_path.parent.mkdir(parents=True, exist_ok=True)
    kg.save(str(kg_path))
    logger.info("KG saved to %s", kg_path)
    return kg


# ---------------------------------------------------------------------------
# Step 5 — Query distribution
# ---------------------------------------------------------------------------


def build_query_distribution(llm, preset: str = "default"):
    """Return a Ragas query_distribution list for the chosen preset.

    Presets:
      - "default":         50% single-hop / 30% multi-hop-specific / 20% multi-hop-abstract
      - "balanced_50_50":  50% single-hop-specific / 50% multi-hop-abstract
      - "semantic_only":   100% multi-hop-abstract (all → expected_function=semantic_general)
    """
    from ragas.testset.synthesizers.multi_hop.abstract import (
        MultiHopAbstractQuerySynthesizer,
    )
    from ragas.testset.synthesizers.multi_hop.specific import (
        MultiHopSpecificQuerySynthesizer,
    )
    from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )

    if preset == "balanced_50_50":
        return [
            (SingleHopSpecificQuerySynthesizer(llm=llm), 0.50),
            (MultiHopAbstractQuerySynthesizer(llm=llm), 0.50),
        ]
    if preset == "semantic_only":
        return [(MultiHopAbstractQuerySynthesizer(llm=llm), 1.0)]
    # default
    return [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 0.50),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 0.30),
        (MultiHopAbstractQuerySynthesizer(llm=llm), 0.20),
    ]


# ---------------------------------------------------------------------------
# Step 6 — Generation
# ---------------------------------------------------------------------------


def generate_testset(kg, llm, embedding_model, query_distribution, testset_size: int):
    """Run RAGAS TestsetGenerator and return a DataFrame."""
    from ragas.testset import TestsetGenerator

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
        knowledge_graph=kg,
    )
    testset = generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution,
        raise_exceptions=False,
    )
    return testset.to_pandas()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Step 7 — Validation
# ---------------------------------------------------------------------------


def validate_testset(df, documents: list[Document], min_ratio: float = 0.7):
    """Drop rows whose reference_contexts cannot be traced to source docs.

    Uses fuzzy string matching against the original document content.
    Returns the filtered DataFrame.
    """
    source_texts = {doc.metadata["source_file"]: doc.page_content for doc in documents}
    all_content = "\n\n".join(source_texts.values())

    keep_mask = []
    for idx, row in df.iterrows():
        contexts = row.get("reference_contexts")
        if not contexts or (isinstance(contexts, list) and len(contexts) == 0):
            logger.debug("Row %s: empty reference_contexts — dropping", idx)
            keep_mask.append(False)
            continue

        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts)
            except json.JSONDecodeError:
                contexts = [contexts]

        matched = False
        for ctx in contexts:
            if not isinstance(ctx, str) or not ctx.strip():
                continue
            ratio = difflib.SequenceMatcher(None, ctx[:500], all_content).quick_ratio()
            if ratio >= min_ratio:
                matched = True
                break
            # Fall back to substring check for short contexts
            if ctx.strip()[:100] in all_content:
                matched = True
                break

        keep_mask.append(matched)

    before = len(df)
    df_clean = df[keep_mask].reset_index(drop=True)
    dropped = before - len(df_clean)
    logger.info("Validation: kept %d / %d items (dropped %d)", len(df_clean), before, dropped)
    return df_clean


# ---------------------------------------------------------------------------
# Step 8 — Export
# ---------------------------------------------------------------------------


def export_golden_set(df, documents: list[Document], output_path: Path) -> None:
    """Export validated testset to XLSX matching golden_set schema."""
    import openpyxl

    # Build source_file → crn lookup
    crn_lookup: dict[str, str] = {}
    for doc in documents:
        crn_lookup[doc.metadata["source_file"]] = doc.metadata.get("crn", "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active

    columns = [
        "id",
        "question",
        "reference_answer",
        "expected_function",
        "human_notes",
        "reference_contexts",
        "crn",
    ]
    ws.append(columns)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        synthesizer_name = row.get("synthesizer_name", "")
        expected_function = SYNTHESIZER_TO_FUNCTION.get(synthesizer_name, "hybrid_course")

        # Serialize reference_contexts
        ref_contexts = row.get("reference_contexts")
        if isinstance(ref_contexts, list):
            ref_contexts_str = json.dumps(ref_contexts, ensure_ascii=False)
        elif ref_contexts is not None:
            ref_contexts_str = str(ref_contexts)
        else:
            ref_contexts_str = ""

        # Try to extract CRN from reference contexts or metadata
        crn = ""
        if ref_contexts and isinstance(ref_contexts, list):
            for ctx in ref_contexts:
                if isinstance(ctx, str):
                    for src_file, src_crn in crn_lookup.items():
                        # Check if this context came from this source doc
                        doc_content = next(
                            (d.page_content for d in documents if d.metadata["source_file"] == src_file),
                            "",
                        )
                        if ctx.strip()[:100] in doc_content:
                            crn = src_crn
                            break
                if crn:
                    break

        ws.append(
            [
                i,  # id
                row.get("user_input", ""),  # question
                row.get("reference", ""),  # reference_answer
                expected_function,  # expected_function
                "",  # human_notes
                ref_contexts_str,  # reference_contexts
                crn,  # crn
            ]
        )

    wb.save(output_path)
    logger.info("Golden set exported to %s (%d items)", output_path, len(df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate RAGAS testset from syllabus eval corpus",
    )
    p.add_argument(
        "--corpus-dir",
        required=True,
        type=Path,
        help="Directory containing v3_step3_flat JSON files",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output XLSX path (default: tamu_data/evals/golden_sets/ragas_{date}.xlsx)",
    )
    # Legacy: kept for back-compat. If set and --target-size is the default, it
    # acts as the target. Otherwise --target-size wins.
    p.add_argument(
        "--testset-size",
        type=int,
        default=None,
        help="(Deprecated alias for --target-size)",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=20,
        help="Final validated item count desired (default: 20)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Items generated per batch before validation+save (default: 5)",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Safety cap on number of generation batches (default: 10)",
    )
    p.add_argument(
        "--distribution",
        choices=list(DISTRIBUTION_PRESETS),
        default="default",
        help="Query distribution preset (default: 'default' = 50/30/20)",
    )
    p.add_argument(
        "--include-terms",
        type=str,
        default=None,
        help="Comma-separated term codes to include, e.g. 202611,202621,202641. If unset, all terms load.",
    )
    p.add_argument(
        "--kg-path",
        type=Path,
        default=Path("tamu_data/evals/ragas_kg.json"),
        help="KG cache file path (default: tamu_data/evals/ragas_kg.json)",
    )
    p.add_argument(
        "--rebuild-kg",
        action="store_true",
        help="Force KG rebuild even if cache exists",
    )
    p.add_argument(
        "--build-kg-only",
        action="store_true",
        help="Build/refresh KG, print stats, and exit before any question generation",
    )
    p.add_argument(
        "--provider",
        choices=["google", "tamu"],
        default="google",
        help="LLM provider (default: google)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="LLM temperature (default: 0.4)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load docs and print stats without generating",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = parse_args()

    # Resolve target size: --target-size wins; --testset-size kept as legacy alias.
    target_size = args.target_size
    if args.testset_size is not None:
        target_size = args.testset_size
        print(f"[deprecation] --testset-size is an alias; using target_size={target_size}")

    include_terms: set[str] | None = None
    if args.include_terms:
        include_terms = {t.strip() for t in args.include_terms.split(",") if t.strip()}
        print(f"Filtering corpus to terms: {sorted(include_terms)}")

    # --- Load corpus ---
    documents = load_corpus(args.corpus_dir, include_terms=include_terms)
    print(f"\nCorpus: {len(documents)} documents from {args.corpus_dir}")

    if not documents:
        print("ERROR: No documents with chunks found. Check --corpus-dir path.")
        return

    token_counts = []
    try:
        from ragas.utils import num_tokens_from_string

        token_counts = [num_tokens_from_string(d.page_content) for d in documents]
        total = sum(token_counts)
        print(
            f"Token stats: total={total:,}, mean={total // len(documents):,}, "
            f"min={min(token_counts):,}, max={max(token_counts):,}"
        )
    except ImportError:
        print(f"Characters: total={sum(len(d.page_content) for d in documents):,}")

    for doc in documents:
        m = doc.metadata
        print(f"  {m['course_id']:>12s}  CRN={m['crn'] or '?':>6s}  {m['source_file']}")

    if args.dry_run:
        print("\n--dry-run: stopping before generation.")
        return

    # --- Build LLM + embeddings ---
    print(f"\nSetting up LLM (provider={args.provider}, temp={args.temperature})...")
    llm = build_llm(args.provider, args.temperature)

    print("Setting up embeddings (Google text-embedding-004)...")
    embedding_model = build_embeddings()

    # --- Build or load KG ---
    kg = build_or_load_kg(
        documents,
        llm,
        embedding_model,
        args.kg_path,
        args.rebuild_kg,
    )

    # KG stats summary
    try:
        n_nodes = len(kg.nodes)
        n_rels = len(kg.relationships)
        node_types: dict[str, int] = {}
        for n in kg.nodes:
            t = getattr(n.type, "value", str(n.type))
            node_types[t] = node_types.get(t, 0) + 1
        print(f"\nKG: nodes={n_nodes} ({node_types}), relationships={n_rels}")
    except Exception as e:  # noqa: BLE001
        print(f"(could not introspect KG: {e})")

    if args.build_kg_only:
        print(f"\n--build-kg-only: stopping after KG build. Cache at {args.kg_path}.")
        return

    # --- Query distribution ---
    query_distribution = build_query_distribution(llm, preset=args.distribution)
    print(f"Distribution preset: {args.distribution}")

    # --- Batched generation loop ---
    import pandas as pd

    output_path = args.output or Path(f"tamu_data/evals/golden_sets/ragas_{datetime.now():%Y%m%d}.xlsx")
    collected = pd.DataFrame()
    for batch_num in range(1, args.max_batches + 1):
        if len(collected) >= target_size:
            break
        remaining = target_size - len(collected)
        # Ragas needs at least a few items per call to use multiple synthesizers
        this_batch = max(min(args.batch_size, remaining + 2), 3)
        print(
            f"\n[batch {batch_num}/{args.max_batches}] "
            f"generating {this_batch} (validated so far: {len(collected)}/{target_size})"
        )
        try:
            raw = generate_testset(kg, llm, embedding_model, query_distribution, this_batch)
        except Exception as e:  # noqa: BLE001
            print(f"[batch {batch_num}] generation error: {e}")
            continue

        validated = validate_testset(raw, documents)
        if len(validated) == 0:
            print(f"[batch {batch_num}] no items survived validation, continuing")
            continue

        collected = pd.concat([collected, validated], ignore_index=True)
        # Incremental save (trimmed to target_size if we overshot)
        export_golden_set(collected.head(target_size), documents, output_path)
        print(f"[batch {batch_num}] kept {len(validated)}, total {len(collected)} → saved {output_path}")

    if len(collected) == 0:
        print("ERROR: No items survived validation across all batches. Check KG quality.")
        return

    final = collected.head(target_size)
    export_golden_set(final, documents, output_path)
    print(f"\nDone! Golden set: {output_path}  ({len(final)} items)")


if __name__ == "__main__":
    main()
