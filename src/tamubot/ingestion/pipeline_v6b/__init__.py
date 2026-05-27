"""pipeline_v6b — RAG-Anything-based ingestion track.

See /home/claude/.claude/plans/toasty-weaving-taco.md for the design.
Runs alongside pipeline_v5 (Docling) and pipeline_v6 (Gemini VLM) and writes
to the same chunks_v4 Atlas collection under chunk_tag in {"v6b_semantic",
"v6b_block"}.
"""
