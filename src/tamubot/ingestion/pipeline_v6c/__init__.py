"""pipeline_v6c — opendataloader-pdf as the bronze extractor.

Additive bake-off track parallel to v6 (Gemini VLM) and v6b (Docling + MinerU).
Bronze stage only; downstream stages reuse the v6 schema (markdown +
HeadersSidecar) so chunkers and evals can compare apples-to-apples.
"""
