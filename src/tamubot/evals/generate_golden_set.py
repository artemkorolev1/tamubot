"""Golden test set synthesis for TamuBot evaluation.

Generates 50 stratified questions from live MongoDB chunks and exports to:
  - tamu_data/logs/golden_set.jsonl  (local JSONL, timestamped + canonical)
  - Langfuse dataset 'tamubot_golden_v1' (via REST API)

Each golden item carries ground truth for ALL pipeline stages:

  Router stage:
    expected_function          — one of 4 functions: hybrid_course, recurrent, semantic_general, out_of_scope
    expected_course_ids        — [] for no-category strata / semantic_general
    expected_specific_categories — [] for default/general strata
    expected_semantic_intent   — True for recurrent, hybrid_course_advisory, semantic_general
    expected_recurrent_search  — True only for recurrent stratum

  Retrieval stage:
    source_crn                 — CRN of the chunk that grounded the question
    source_category            — syllabus category of that chunk
    source_course_id           — course the chunk belongs to
    (Reranker ordering is evaluated via automated NDCG in eval_retrieval_metrics.py
     rather than hand-labelled ranks — no additional field needed here.)

  Generator stage:
    reference_answer           — chunk content the answer should be grounded in

Category sampling is weighted by real student query priority.
Function type emerges from how the question is phrased (framing prompt),
not from which category it's about.

Usage:
    python scripts/generate_golden_set.py                    # full 50-question run
    python scripts/generate_golden_set.py --dry-run          # validate distribution only
    python scripts/generate_golden_set.py --dry-run --n 10   # show scaled distribution
    python scripts/generate_golden_set.py --no-langfuse      # skip Langfuse upload
    python scripts/generate_golden_set.py --department CSCE
    python scripts/generate_golden_set.py --seed 123         # reproducible run
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Category importance weights
#
# Reflects what students actually query — not internal routing concerns.
# GRADING and ATTENDANCE_AND_MAKEUP are useful but over-represented in
# typical eval sets; halved here so higher-signal categories dominate.
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS: dict[str, float] = {
    "COURSE_OVERVIEW": 0.20,  # what is this course, topics, description
    "LEARNING_OUTCOMES": 0.17,  # what will I learn, skills, objectives
    "PREREQUISITES": 0.14,  # required background, can I take this
    "SCHEDULE": 0.14,  # exam dates, assignment deadlines, lecture topics
    "MATERIALS": 0.10,  # textbooks, tools, software, cost
    "GRADING": 0.10,  # halved from initial weight — useful but over-queried
    "AI_POLICY": 0.07,  # ChatGPT, generative AI rules — high student interest
    "INSTRUCTOR": 0.04,  # office hours, contact info, TA details
    "ATTENDANCE_AND_MAKEUP": 0.02,  # halved — late work policy, rarely a primary need
    "UNIVERSITY_POLICIES": 0.01,  # boilerplate, rarely a primary query target
    "SUPPORT_SERVICES": 0.01,  # disability services, counseling
}

# Normalized probabilities (computed once at import)
_TOTAL_WEIGHT = sum(CATEGORY_WEIGHTS.values())
CATEGORY_PROBS: dict[str, float] = {cat: w / _TOTAL_WEIGHT for cat, w in CATEGORY_WEIGHTS.items()}


def weighted_sample_categories(n: int, rng: random.Random | None = None) -> list[str]:
    """Draw n category names with replacement according to CATEGORY_WEIGHTS."""
    rng = rng or random.Random()
    cats = list(CATEGORY_PROBS.keys())
    probs = list(CATEGORY_PROBS.values())
    return rng.choices(cats, weights=probs, k=n)


# ---------------------------------------------------------------------------
# Function strata
#
# Each stratum defines: how many questions, a framing prompt for synthesis,
# and whether questions should name a specific course ID.
#
# No-category strata (metadata_default, metadata_default_advisory, semantic_general, recurrent_default)
# make up ~64% of the set — broad queries that reflect how students actually
# approach a course assistant before diving into specifics.
#
# Category-specific strata (metadata_specific, metadata_specific_evaluative, metadata_combined)
# draw from the weighted pool above so high-value categories dominate.
# ---------------------------------------------------------------------------

STRATUM_MAP: dict[str, dict] = {
    # ── hybrid_course strata (54%) ────────────────────────────────────
    "hybrid_course_default": {
        "expected_function": "hybrid_course",
        "n_questions": 14,
        "has_category": False,
        "expected_semantic_intent": False,
        "expected_recurrent_search": False,
        "description": "General overview — no specific category → hybrid_course",
        "framing": (
            "Ask a broad, general question about the course that does NOT focus on "
            "any specific syllabus category. Use the course ID from the excerpt. "
            "Do NOT ask for opinions or difficulty assessments. "
            "Examples: 'Tell me about CSCE 638', 'What is CSCE 670 about?', "
            "'Give me an overview of this course', 'What should I know before enrolling?'"
        ),
        "use_course_id": True,
    },
    "hybrid_course_advisory": {
        "expected_function": "hybrid_course",
        "n_questions": 8,
        "has_category": False,
        "expected_semantic_intent": True,
        "expected_recurrent_search": False,
        "description": "Advisory/evaluative question about a known course → hybrid_course",
        "framing": (
            "Ask a subjective or advisory question about the course WITHOUT naming "
            "a specific syllabus category. Use evaluative or career-oriented language. "
            "Use the course ID from the excerpt. "
            "Examples: 'Is this course worth taking?', 'How hard is CSCE 638?', "
            "'Is this good for a career in data science?', "
            "'Should I take this before CSCE 670?', 'Would you recommend this course?'"
        ),
        "use_course_id": True,
    },
    "hybrid_course_specific": {
        "expected_function": "hybrid_course",
        "n_questions": 10,
        "has_category": True,
        "expected_semantic_intent": False,
        "expected_recurrent_search": False,
        "description": "Specific category question, specific_only=True → hybrid_course",
        "framing": (
            "Ask a direct factual question about the specific category shown in the excerpt. "
            "The question must clearly imply or name this category. "
            "Do NOT ask for opinions or difficulty assessments. "
            "Use the course ID from the excerpt. "
            "Examples for GRADING: 'What is the grading breakdown for CSCE 638?' "
            "Examples for SCHEDULE: 'When is the midterm for CSCE 638?' "
            "Examples for PREREQUISITES: 'What are the prerequisites for CSCE 638?' "
            "Examples for MATERIALS: 'What textbooks are required for CSCE 670?'"
        ),
        "use_course_id": True,
    },
    "hybrid_course_combined": {
        "expected_function": "hybrid_course",
        "n_questions": 5,
        "has_category": True,
        "expected_semantic_intent": False,
        "expected_recurrent_search": False,
        "description": "Overview with category emphasis, specific_only=False → hybrid_course",
        "framing": (
            "Ask a general question about the course that mentions the category from "
            "the excerpt as background context — not as the exclusive focus. "
            "Use phrases like 'especially', 'particularly', 'with a focus on', 'including'. "
            "Use the course ID from the excerpt. "
            "Examples: 'Tell me about CSCE 638, especially the grading', "
            "'Give me an overview of CSCE 670 with a focus on learning outcomes', "
            "'What should I know about CSCE 638, including its AI policy?'"
        ),
        "use_course_id": True,
    },
    # ── recurrent (10%) ───────────────────────────────────────────────
    "recurrent": {
        "expected_function": "recurrent",
        "n_questions": 5,
        "has_category": False,
        "expected_semantic_intent": True,
        "expected_recurrent_search": True,
        "description": "Course discovery/pairing — anchor course, seeking complementary courses",
        "framing": (
            "Ask a question that seeks to find OTHER courses that pair with, follow, or "
            "complement the course in the excerpt. Use the course ID from the excerpt as the anchor. "
            "Do NOT name a second course — the user is looking for unknown courses. "
            "Examples: 'What course should I take with CSCE 638?', "
            "'What courses are similar to CSCE 670?', "
            "'What should I take after CSCE 638?', "
            "'What goes well with CSCE 638?'"
        ),
        "use_course_id": True,
    },
    # ── semantic_general (16%) ────────────────────────────────────────
    "semantic_general": {
        "expected_function": "semantic_general",
        "n_questions": 8,
        "has_category": False,
        "expected_semantic_intent": True,
        "expected_recurrent_search": False,
        "description": "Discovery question — no specific course ID",
        "framing": (
            "Ask a discovery, cross-course, or advisory question about TAMU academics "
            "WITHOUT mentioning any specific course ID. The question should be answerable "
            "by searching across courses or using general academic knowledge. "
            "Examples: 'Which TAMU CS courses cover machine learning?', "
            "'What courses are good preparation for a PhD in AI?', "
            "'What are the typical grading norms in TAMU graduate CS courses?', "
            "'Which courses have the lightest workload in CSCE?'"
        ),
        "use_course_id": False,
    },
    # ── out_of_scope (4%) ─────────────────────────────────────────────
    "out_of_scope": {
        "expected_function": "out_of_scope",
        "n_questions": 2,
        "has_category": False,
        "expected_semantic_intent": False,
        "expected_recurrent_search": False,
        "description": "Off-topic — not about TAMU academics",
        "framing": None,
        "use_course_id": False,
    },
}

# Total: 14+8+10+5+5+8+2 = 52 — trimmed to 50 by generate_golden_set's all_questions[:n_total]

# Synthetic off-topic questions — drawn randomly for out_of_scope
OUT_OF_SCOPE_QUESTIONS: list[str] = [
    "What's the weather forecast for College Station this weekend?",
    "Can you write a Python function to sort a list?",
    "What are the best restaurants near the TAMU campus?",
    "Write me a cover letter for a software engineering internship at Google.",
    "What is the capital of France?",
    "How do I set up a React project from scratch?",
    "What's a good recipe for chicken tikka masala?",
    "Who won the Super Bowl last year?",
]


# ---------------------------------------------------------------------------
# Derive router ground truth from stratum + chunk
# ---------------------------------------------------------------------------


def _derive_router_ground_truth(
    stratum: str,
    chunk: dict,
) -> dict:
    """Derive expected router output fields from stratum type and source chunk.

    These are the "correct answers" the router should produce for this question —
    used by the benchmark to measure router accuracy.
    """
    spec = STRATUM_MAP[stratum]
    course_id = chunk.get("course_id", "")

    # expected_course_ids: present for all strata that use a course ID
    if spec.get("use_course_id") and course_id:
        expected_course_ids = [course_id]
    else:
        expected_course_ids = []

    # expected_specific_categories: chunks no longer carry a category field
    expected_specific_categories: list[str] = []

    return {
        "expected_function": spec.get("expected_function", stratum),
        "expected_course_ids": expected_course_ids,
        "expected_specific_categories": expected_specific_categories,
        "expected_semantic_intent": spec["expected_semantic_intent"],
        "expected_recurrent_search": spec.get("expected_recurrent_search", False),
    }


# ---------------------------------------------------------------------------
# MongoDB chunk sampling
# ---------------------------------------------------------------------------


def sample_chunks_from_mongo(
    n_total: int,
    department: str = "CSCE",
) -> list[dict]:
    """Sample chunks from MongoDB.

    Chunks no longer carry a category field, so sampling is uniform over the
    department's chunk pool. Caller is responsible for downstream stratification.
    """
    from pymongo import MongoClient

    from tamubot.core import config

    client: MongoClient = MongoClient(config.MONGODB_URI)
    db = client[config.MONGODB_DB]
    chunks_col = db["chunks"]

    pipeline = [
        {
            "$match": {
                "course_id": {"$regex": f"^{department}", "$options": "i"},
                "content": {"$exists": True, "$ne": ""},
            }
        },
        {"$sample": {"size": n_total}},
        {
            "$project": {
                "_id": 0,
                "crn": 1,
                "course_id": 1,
                "title": 1,
                "content": 1,
                "section": 1,
                "term": 1,
                "instructor_name": 1,
            }
        },
    ]
    sampled_chunks = list(chunks_col.aggregate(pipeline))  # type: ignore[arg-type]
    client.close()
    print(f"  Total sampled: {len(sampled_chunks)} chunks")
    return sampled_chunks


# ---------------------------------------------------------------------------
# Question synthesis
# ---------------------------------------------------------------------------


def synthesize_question_for_chunk(
    chunk: dict,
    stratum: str,
    framing: str,
    use_course_id: bool,
) -> dict | None:
    """Synthesize one realistic student question from a chunk + framing prompt.

    Returns an enriched question dict with ground truth for all pipeline stages,
    or None on synthesis failure.
    """
    from google.genai import types

    from tamubot.core import config

    client = config.get_genai_client()
    course_id = chunk.get("course_id", "the course")
    content = chunk.get("content", "")[:700]
    crn = chunk.get("crn", "")

    course_hint = (
        f"The excerpt is from course {course_id} (CRN: {crn})."
        if use_course_id
        else "Do NOT mention a specific course ID in your question."
    )

    prompt = f"""\
You are generating realistic student evaluation questions for a TAMU course assistant.

Syllabus excerpt:
---
{content}
---

{course_hint}

Task: Generate ONE realistic student question grounded in this excerpt.

Question type guidance:
{framing}

Rules:
- The question must be directly answerable using this excerpt
- Use natural student language (not formal/academic)
- One sentence, ending with a question mark

Respond with ONLY the question text, nothing else.
"""

    try:
        if config.USE_TAMU_API:
            tamu = config.get_tamu_client()
            stream = tamu.chat.completions.create(
                model=config.TAMU_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=4096,
                stream=True,
            )
            raw = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
        else:
            resp = client.models.generate_content(
                model=config.GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.8, max_output_tokens=120),
            )
            raw = resp.text
        question = raw.strip().strip('"').strip("'")
        if not question.endswith("?"):
            question += "?"

        router_gt = _derive_router_ground_truth(stratum, chunk)

        return {
            # Question
            "question": question,
            # Router ground truth
            "expected_function": router_gt["expected_function"],
            "expected_course_ids": router_gt["expected_course_ids"],
            "expected_specific_categories": router_gt["expected_specific_categories"],
            "expected_semantic_intent": router_gt["expected_semantic_intent"],
            # Retrieval ground truth
            # The source chunk IS the relevant item — recall@k checks whether
            # it (or another chunk from the same course) surfaces in results.
            "source_crn": crn,
            "source_course_id": course_id,
            # Generator ground truth
            "reference_answer": content,
            # Metadata
            "stratum": stratum,
            "source": "gemini_synthesis",
        }

    except Exception as e:
        print(f"  [WARN] Synthesis failed for {course_id}: {e}")
        return None


def synthesize_stratum(
    stratum: str,
    spec: dict,
    all_chunks: list[dict],
    n_questions: int,
    rng: random.Random,
) -> list[dict]:
    """Synthesize n_questions for a stratum, sampling chunks uniformly."""
    framing = spec["framing"]
    use_course_id = spec.get("use_course_id", True)

    if not all_chunks:
        print(f"  [WARN] No chunks available for stratum '{stratum}'")
        return []

    results: list = []
    attempts = 0
    max_attempts = n_questions * 4

    while len(results) < n_questions and attempts < max_attempts:
        attempts += 1
        chunk = rng.choice(all_chunks)
        item = synthesize_question_for_chunk(chunk, stratum, framing, use_course_id)
        if item:
            results.append(item)
            print(f"  [{len(results):2d}/{n_questions}] {item['question'][:65]}")

    if len(results) < n_questions:
        print(f"  [WARN] Only synthesized {len(results)}/{n_questions} for '{stratum}'")

    return results


# ---------------------------------------------------------------------------
# Langfuse dataset upload
# ---------------------------------------------------------------------------


def upload_to_langfuse(
    questions: list[dict],
    dataset_name: str = "tamubot_golden_v1",
) -> bool:
    """Create (or upsert) a Langfuse dataset and upload all items."""
    import httpx

    from tamubot.core import config

    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        print("  [WARN] Langfuse credentials not configured — skipping upload.")
        return False

    host = config.LANGFUSE_BASE_URL.rstrip("/")
    auth = (config.LANGFUSE_PUBLIC_KEY, config.LANGFUSE_SECRET_KEY)

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{host}/api/public/datasets",
                json={
                    "name": dataset_name,
                    "description": (
                        "TamuBot golden eval set — questions grounded in real syllabus chunks, "
                        "weighted by student query priority. Ground truth for router, retrieval, "
                        "and generator pipeline stages."
                    ),
                },
                auth=auth,
            )
            if resp.status_code not in (200, 201):
                print(f"  [WARN] Dataset creation: HTTP {resp.status_code}")
            else:
                print(f"  Langfuse dataset '{dataset_name}' ready.")

            uploaded = 0
            for q in questions:
                payload = {
                    "datasetName": dataset_name,
                    "input": {"question": q["question"]},
                    "expectedOutput": {
                        # Router
                        "expected_function": q.get("expected_function"),
                        "expected_course_ids": q.get("expected_course_ids", []),
                        "expected_specific_categories": q.get("expected_specific_categories", []),
                        "expected_semantic_intent": q.get("expected_semantic_intent", False),
                        # Retrieval
                        "source_crn": q.get("source_crn"),
                        "source_course_id": q.get("source_course_id"),
                        # Generator
                        "reference_answer": q.get("reference_answer", ""),
                    },
                    "metadata": {
                        "stratum": q.get("stratum"),
                        "source": q.get("source"),
                    },
                }
                r = client.post(f"{host}/api/public/dataset-items", json=payload, auth=auth)
                if r.status_code in (200, 201):
                    uploaded += 1
                else:
                    print(f"  [WARN] Item upload failed: HTTP {r.status_code}")

        print(f"  Langfuse: {uploaded}/{len(questions)} items uploaded.")
        return uploaded > 0

    except Exception as e:
        print(f"  [ERROR] Langfuse upload failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_golden_set(
    n_total: int = 50,
    dry_run: bool = False,
    no_langfuse: bool = False,
    department: str = "CSCE",
    output_dir: Path = Path("tamu_data/logs"),
    seed: int = 42,
    tag: str = "",
) -> list[dict]:
    """Full golden set generation pipeline."""
    rng = random.Random(seed)

    # Scale per-stratum counts if n_total != 50
    base_total = sum(s["n_questions"] for s in STRATUM_MAP.values())
    if n_total != base_total:
        scale = n_total / base_total
        stratum_counts = {k: max(1, round(v["n_questions"] * scale)) for k, v in STRATUM_MAP.items()}
    else:
        stratum_counts = {k: v["n_questions"] for k, v in STRATUM_MAP.items()}

    no_cat_n = sum(
        stratum_counts[s] for s in ("hybrid_course_default", "hybrid_course_advisory", "semantic_general", "recurrent")
    )
    with_cat_n = sum(stratum_counts[s] for s in ("hybrid_course_specific", "hybrid_course_combined"))

    print(f"\n{'=' * 60}")
    print("  TamuBot Golden Set Generation")
    print(f"  Target: {n_total} q  |  Dept: {department}  |  Seed: {seed}")
    print(
        f"  No-category: {no_cat_n} ({no_cat_n / n_total:.0%})  "
        f"Category-specific: {with_cat_n} ({with_cat_n / n_total:.0%})  "
        f"OOS: {stratum_counts['out_of_scope']}"
    )
    print(f"{'=' * 60}")

    # ── Step 1: Sample chunks ──────────────────────────────────────────
    n_retrieval_q = sum(v for k, v in stratum_counts.items() if k != "out_of_scope")
    n_chunks = n_retrieval_q * 4  # 4x oversample for variety + retry headroom

    print(f"\n[1/4] Sampling ~{n_chunks} chunks from MongoDB...")
    if dry_run:
        print("  DRY-RUN: skipping MongoDB calls.")
        all_chunks: list[dict] = []
    else:
        all_chunks = sample_chunks_from_mongo(n_total=n_chunks, department=department)

    # ── Step 2: Synthesize per stratum ────────────────────────────────
    print("\n[2/4] Synthesizing questions...")
    all_questions: list[dict] = []

    for stratum, spec in STRATUM_MAP.items():
        if stratum == "out_of_scope":
            continue
        n_q = stratum_counts[stratum]
        has_cat = spec["has_category"]
        sem = spec["expected_semantic_intent"]
        print(f"\n  [{stratum}]  n={n_q}  category={'yes' if has_cat else 'no'}  semantic_intent={sem}")
        print(f"  {spec['description']}")

        if dry_run:
            print(f"  DRY-RUN: would synthesize {n_q} questions.")
            continue

        questions = synthesize_stratum(stratum, spec, all_chunks, n_q, rng)
        all_questions.extend(questions)

    # ── Step 3: out_of_scope ───────────────────────────────────────────
    n_oos = stratum_counts["out_of_scope"]
    oos_items: list = [
        {
            "question": q,
            "expected_function": "out_of_scope",
            "expected_course_ids": [],
            "expected_specific_categories": [],
            "expected_semantic_intent": False,
            "source_crn": None,
            "source_course_id": None,
            "reference_answer": "(out of scope)",
            "stratum": "out_of_scope",
            "source": "synthetic_oos",
        }
        for q in rng.sample(OUT_OF_SCOPE_QUESTIONS, min(n_oos, len(OUT_OF_SCOPE_QUESTIONS)))
    ]

    print(f"\n[3/4] Adding {n_oos} out_of_scope questions (synthetic).")
    if not dry_run:
        all_questions.extend(oos_items)

    if dry_run:
        print("\n  DRY-RUN complete. Planned distribution:")
        print(f"  {'Stratum':<22} {'n':>4}  {'Category':>8}  {'semantic_intent':>16}")
        print(f"  {'-' * 56}")
        for s, spec in STRATUM_MAP.items():
            n = stratum_counts[s]
            has_c = "yes" if spec["has_category"] else "no"
            sem = str(spec["expected_semantic_intent"])
            print(f"  {s:<22} {n:>4}  {has_c:>8}  {sem:>16}")
        total_no_cat = sum(stratum_counts[s] for s, sp in STRATUM_MAP.items() if not sp["has_category"])
        total_cat = sum(stratum_counts[s] for s, sp in STRATUM_MAP.items() if sp["has_category"])
        print(f"\n  No-category queries: {total_no_cat}/{n_total} ({total_no_cat / n_total:.0%})")
        print(f"  Category-specific:   {total_cat}/{n_total} ({total_cat / n_total:.0%})")
        print("\n  Category weights (descending):")
        for cat, w in sorted(CATEGORY_WEIGHTS.items(), key=lambda x: -x[1]):
            bar = "#" * round(w * 100)
            print(f"    {cat:<30} {w:.0%}  {bar}")
        return []

    # Trim to n_total
    all_questions = all_questions[:n_total]

    # ── Step 4: Export ─────────────────────────────────────────────────
    print(f"\n[4/4] Exporting {len(all_questions)} questions...")

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"golden_set_{tag}" if tag else "golden_set"
    timestamped_path = output_dir / f"{prefix}_{ts}.jsonl"
    canonical_path = output_dir / f"{prefix}.jsonl"

    for path in (timestamped_path, canonical_path):
        with path.open("w", encoding="utf-8") as f:
            for q in all_questions:
                f.write(json.dumps(q) + "\n")
    print(f"  Timestamped: {timestamped_path}")
    print(f"  Canonical:   {canonical_path}")

    if not no_langfuse:
        print("  Uploading to Langfuse dataset 'tamubot_golden_v1'...")
        upload_to_langfuse(all_questions)
    else:
        print("  Langfuse upload skipped (--no-langfuse).")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary: {len(all_questions)} questions generated")
    print(f"{'=' * 60}")

    print("\n  By stratum:")
    strata_counts: dict[str, int] = {}
    for q in all_questions:
        s = q.get("stratum", "?")
        strata_counts[s] = strata_counts.get(s, 0) + 1
    for s in STRATUM_MAP:
        n = strata_counts.get(s, 0)
        spec = STRATUM_MAP[s]
        tag = "(no cat)" if not spec["has_category"] else "(cat)"
        print(f"    {s:<22} {n:2d}  {tag}")

    return all_questions


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate a stratified golden test set for TamuBot evaluation")
    parser.add_argument("--n", type=int, default=50, help="Target question count (default: 50)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show planned distribution only — skip synthesis and export"
    )
    parser.add_argument("--no-langfuse", action="store_true", help="Skip Langfuse dataset upload")
    parser.add_argument("--department", default="CSCE", help="MongoDB department filter (default: CSCE)")
    parser.add_argument(
        "--output-dir", default="tamu_data/logs", help="Local output directory (default: tamu_data/logs)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument(
        "--tag", default="", help="Tag appended to output filename, e.g. 'csce' → golden_set_csce.jsonl"
    )
    args = parser.parse_args()

    generate_golden_set(
        n_total=args.n,
        dry_run=args.dry_run,
        no_langfuse=args.no_langfuse,
        department=args.department,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
