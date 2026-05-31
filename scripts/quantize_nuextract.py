"""Offline W4A16 quantization of NuExtract3 via llm-compressor, for vLLM serving.

Run in a vLLM/llm-compressor environment (NOT the dev container's torch 2.12 —
run it inside the sidecar image or a dedicated env). Produces a compressed-tensors
checkpoint vLLM auto-detects. Calibrates on real syllabus markdown so the
quantization sees in-domain text.

Usage (host / sidecar env):
    python scripts/quantize_nuextract.py --out models/NuExtract3-W4A16 --calib 64
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "numind/NuExtract3"

# Inlined from tamubot.ingestion.clients.nuextract_client.SYLLABUS_TEMPLATE — duplicated
# deliberately so this quantization tool (run in a minimal llm-compressor container)
# does NOT import the RAG app's dependency tree (langgraph/langchain/etc.). Keep in sync.
SYLLABUS_TEMPLATE = {
    "course_code": "verbatim-string",
    "course_title": "verbatim-string",
    "instructor_name": "verbatim-string",
    "instructor_email": "verbatim-string",
    "credit_hours": "integer",
    "meeting_schedule": [{"day": "verbatim-string", "time": "verbatim-string", "location": "verbatim-string"}],
    "assessment_weights": [{"component": "verbatim-string", "weight_pct": "number"}],
    "letter_grade_cutoffs": [{"grade": "verbatim-string", "min_percent": "number"}],
    "prerequisites": ["verbatim-string"],
    "learning_outcomes": ["verbatim-string"],
    "attendance_policy": "verbatim-string",
    "academic_integrity_policy": "verbatim-string",
}

# DATA_ROOT is data/syllabi; bronze markdown lives at <DATA_ROOT>/<DEPT>/v6b/bronze/*.md.
DEFAULT_BRONZE_GLOB = "data/syllabi/**/v6b/bronze/*.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/NuExtract3-W4A16")
    ap.add_argument("--calib", type=int, default=64, help="number of calibration docs")
    ap.add_argument("--bronze-glob", default=DEFAULT_BRONZE_GLOB)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tmpl = json.dumps(SYLLABUS_TEMPLATE, indent=4)

    # Load the VLM explicitly (llm-compressor's default loader is AutoModelForCausalLM,
    # which rejects this image-text model). CPU load + bf16; GPTQ's sequential pipeline
    # onloads one layer at a time to the GPU, so the full model never sits on the 8GB card.
    # This is also the load gate: if trust_remote_code modeling fails under the installed
    # transformers, it surfaces here clearly before any calibration work.
    print(f"[load] loading {MODEL_ID} on CPU (bf16)...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] model loaded: {n_params / 1e9:.2f}B params", flush=True)

    files = sorted(glob.glob(args.bronze_glob, recursive=True))[: args.calib]
    assert files, f"no calibration md found at {args.bronze_glob}"
    texts = []
    for f in files:
        md = Path(f).read_text(encoding="utf-8")
        texts.append(
            processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": md}]}],
                add_generation_prompt=True,
                tokenize=False,
                template=tmpl,
                enable_thinking=False,
            )
        )
    ds = Dataset.from_dict({"text": texts})

    # W4A16 GPTQ on Linear layers. lm_head and the vision tower stay full precision;
    # if oneshot errors on a gated-deltanet module, add its module name here.
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:.*visual.*", "re:.*vision.*"],
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        output_dir=args.out,
        max_seq_length=4096,
        num_calibration_samples=len(texts),
        trust_remote_code_model=True,
    )
    # Persist the processor/tokenizer + custom modeling code alongside the weights so
    # vLLM can load the checkpoint with --trust-remote-code.
    processor.save_pretrained(args.out)
    print(f"[done] quantized checkpoint at {args.out}")


if __name__ == "__main__":
    main()
