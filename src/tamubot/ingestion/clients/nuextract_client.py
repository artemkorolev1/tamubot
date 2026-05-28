"""In-process NuExtract3 client for structured syllabus extraction.

NuExtract3 is a vision-language model (Qwen3.5 base). Extraction is driven by a
custom chat template: the JSON schema goes through
`processor.apply_chat_template(..., template=..., enable_thinking=False)`, NOT a
free-text prompt. We load it at NF4 4-bit (~3-4 GB VRAM) so it fits an 8 GB GPU.

Heavy deps (torch / transformers / PIL) are imported lazily inside the methods
that need them, so `parse_extract` and the template can be unit-tested without a
GPU or the model on disk.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from tamubot.rag.models_v4 import SyllabusExtract

if TYPE_CHECKING:
    from PIL.Image import Image

MODEL_ID = "numind/NuExtract3"

# NuExtract template: leaves are type markers ("verbatim-string", "integer",
# "number"); nested records are single-element lists of dicts. Keys MUST stay in
# sync with SyllabusExtract.model_fields (enforced by test).
SYLLABUS_TEMPLATE: dict[str, Any] = {
    "course_code": "verbatim-string",
    "course_title": "verbatim-string",
    "instructor_name": "verbatim-string",
    "instructor_email": "verbatim-string",
    "credit_hours": "integer",
    "meeting_schedule": [
        {"day": "verbatim-string", "time": "verbatim-string", "location": "verbatim-string"}
    ],
    "assessment_weights": [{"component": "verbatim-string", "weight_pct": "number"}],
    "letter_grade_cutoffs": [{"grade": "verbatim-string", "min_percent": "number"}],
    "prerequisites": ["verbatim-string"],
    "learning_outcomes": ["verbatim-string"],
    "attendance_policy": "verbatim-string",
    "academic_integrity_policy": "verbatim-string",
}

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _strip_json_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def parse_extract(raw: str) -> SyllabusExtract:
    """Parse raw model output into a SyllabusExtract. Tolerates ```json fences
    and surrounding prose. Raises ValueError if no JSON object is present."""
    text = _strip_json_fence(raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON object in model output: {raw[:200]!r}") from None
        data = json.loads(text[start : end + 1])
    return SyllabusExtract.model_validate(data)


class NuExtractExtractor:
    """Holds a loaded NuExtract3 model + processor. Load once, reuse per call.

    Construct via `from_pretrained()` for the real 4-bit model, or pass a
    model/processor directly (e.g. fakes in tests).
    """

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID, *, quantize: bool = True) -> NuExtractExtractor:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if quantize:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = torch.bfloat16
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
        return cls(model, processor)

    def _generate(self, content: list[dict], *, max_new_tokens: int = 4096) -> str:
        import torch

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            template=json.dumps(SYLLABUS_TEMPLATE, indent=4),
            enable_thinking=False,
        ).to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = gen[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def extract_text(self, markdown: str) -> SyllabusExtract:
        return parse_extract(self._generate([{"type": "text", "text": markdown}]))

    def extract_image(self, image: "Image") -> SyllabusExtract:
        return parse_extract(self._generate([{"type": "image", "image": image}]))
