"""Image + Table modal processors adapted from RAG-Anything (HKUDS).

Vendor strategy: prompts lifted verbatim from upstream prompt.py (vision_prompt,
table_prompt, IMAGE_ANALYSIS_SYSTEM, TABLE_ANALYSIS_SYSTEM). JSON-parse + think-
tag strip logic preserved. LightRAG persistence (entity VDB upserts, knowledge
graph merge, chunks VDB upserts) is NOT vendored — these processors return
plain dicts; storage is the caller's responsibility.

Each processor exposes a synchronous `process(...)` returning:

    {"block_id": str, "description": str, "caption": str,
     "entities": list[dict], "raw_response": str, "confidence": float}

`description` is the long-form analysis. `caption` is a short summary suitable
for inline markdown replacement. `entities` are extracted noun phrases (may be
empty). `confidence` is 1.0 on a clean JSON parse, 0.5 on think-tag fallback,
0.0 on hard failure.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

log = logging.getLogger(__name__)


# Prompts lifted verbatim from upstream raganything/prompt.py (commit 8395634).
IMAGE_ANALYSIS_SYSTEM = "You are an expert image analyst. Provide detailed, accurate descriptions."

TABLE_ANALYSIS_SYSTEM = "You are an expert data analyst. Provide detailed table analysis with specific insights."

VISION_PROMPT = """Please analyze this image in detail and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
    - Describe the overall composition and layout
    - Identify all objects, people, text, and visual elements
    - Explain relationships between elements
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - Always use specific names instead of pronouns",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content and its significance (max 100 words)"
    }}
}}

Additional context:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Focus on providing accurate, detailed visual analysis that would be useful for knowledge retrieval."""

TABLE_PROMPT = """Reconstruct the table in the image as github-flavored markdown (GFM).

Return a JSON object with exactly these fields:

{{
    "table_markdown": "the GFM table — header row, separator row (|---|), data rows. Preserve every cell verbatim. No prose, no truncation, no '...'. If the image contains multiple separate tables, concatenate them with one blank line between.",
    "caption": "one short sentence (max 25 words) naming what the table is for",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose (max 30 words)"
    }}
}}

Constraints:
- table_markdown must be valid GFM that renders as a grid.
- Do not invent rows or columns that aren't in the image.
- Do not add analysis or commentary in table_markdown — keep it strictly tabular.
- If a cell is empty, output an empty cell (||), not "N/A" or "-".

Table metadata (for disambiguation only — do not include in the markdown):
Caption: {table_caption}
Footnotes: {table_footnote}
Path: {table_img_path}"""


# Vision LLM callable signature:
#     fn(prompt: str, image_b64: Optional[str], system_prompt: str) -> str
VisionFunc = Callable[[str, Optional[str], str], str]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning-model output."""
    return _THINK_RE.sub("", text).strip()


def _robust_json_parse(response: str) -> Dict[str, Any]:
    """Try multiple strategies to extract a JSON object from an LLM response.

    Order: direct json.loads → strip ``` fences → strip <think> tags → find
    first {...} balanced brace pair. Raises ValueError if nothing parses.
    """
    cleaned = response.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    m = _JSON_FENCE_RE.search(cleaned)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    stripped = _strip_thinking_tags(cleaned)
    if stripped != cleaned:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(stripped)):
            if stripped[i] == "{":
                depth += 1
            elif stripped[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not parse JSON from response: {response[:200]!r}")


def _encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """Read image file and return base64-encoded contents."""
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


class ImageModalProcessor:
    """Vision-LLM image description.

    Construct with a vision callable, then call .process(block) for each
    image block emitted by a Parser. Optional cache writes JSONL keyed on
    image content hash so re-runs are free.
    """

    def __init__(
        self,
        vision_func: VisionFunc,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.vision_func = vision_func
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        if cache_path and cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        with open(self.cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._cache[row["key"]] = row["value"]
                except json.JSONDecodeError, KeyError:
                    continue

    def _persist(self, key: str, value: Dict[str, Any]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "value": value}) + "\n")

    @staticmethod
    def _cache_key(image_b64: str) -> str:
        return "img:" + hashlib.sha256(image_b64.encode()).hexdigest()[:16]

    def process(self, block: Dict[str, Any], block_id: str) -> Dict[str, Any]:
        if block.get("type") != "image":
            raise ValueError(f"ImageModalProcessor expects type='image', got {block.get('type')!r}")
        img_path = block.get("img_path")
        if not img_path:
            return self._fail(block_id, "no img_path on block")
        if not Path(img_path).exists():
            return self._fail(block_id, f"image not found: {img_path}")

        image_b64 = _encode_image_to_base64(img_path)
        cache_key = self._cache_key(image_b64)
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["block_id"] = block_id
            cached["from_cache"] = True
            return cached

        captions = block.get("image_caption") or "None"
        footnotes = block.get("image_footnote") or "None"
        prompt = VISION_PROMPT.format(
            entity_name=f"image_{block_id}",
            image_path=img_path,
            captions=captions,
            footnotes=footnotes,
        )

        try:
            raw = self.vision_func(prompt, image_b64, IMAGE_ANALYSIS_SYSTEM)
        except Exception as exc:
            log.warning("vision call failed for block %s: %s", block_id, exc)
            return self._fail(block_id, str(exc))

        result = self._parse(raw, block_id)
        self._cache[cache_key] = result
        self._persist(cache_key, result)
        return result

    def _parse(self, raw: str, block_id: str) -> Dict[str, Any]:
        try:
            data = _robust_json_parse(raw)
            description = data.get("detailed_description") or ""
            entity = data.get("entity_info") or {}
            caption = entity.get("summary") or description[:200]
            return {
                "block_id": block_id,
                "description": description,
                "caption": caption,
                "entities": [entity] if entity else [],
                "raw_response": raw,
                "confidence": 1.0,
            }
        except ValueError as exc:
            log.info("json parse failed for image block %s: %s", block_id, exc)
            fallback = _strip_thinking_tags(raw)
            return {
                "block_id": block_id,
                "description": fallback,
                "caption": fallback[:200],
                "entities": [],
                "raw_response": raw,
                "confidence": 0.5,
            }

    @staticmethod
    def _fail(block_id: str, reason: str) -> Dict[str, Any]:
        return {
            "block_id": block_id,
            "description": "",
            "caption": f"[image processing failed: {reason}]",
            "entities": [],
            "raw_response": "",
            "confidence": 0.0,
        }


class TableModalProcessor:
    """Vision-LLM table description.

    Like ImageModalProcessor but for table blocks. When the block has an
    img_path the vision model sees the rendered table; when img_path is empty
    it falls back to a text-only call with the markdown body in the prompt.
    """

    def __init__(
        self,
        vision_func: VisionFunc,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.vision_func = vision_func
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        if cache_path and cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        with open(self.cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._cache[row["key"]] = row["value"]
                except json.JSONDecodeError, KeyError:
                    continue

    def _persist(self, key: str, value: Dict[str, Any]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "value": value}) + "\n")

    @staticmethod
    def _cache_key(body_repr: str, image_b64: str) -> str:
        h = hashlib.sha256((body_repr + "|" + image_b64).encode()).hexdigest()[:16]
        return "tbl:" + h

    def process(self, block: Dict[str, Any], block_id: str) -> Dict[str, Any]:
        if block.get("type") != "table":
            raise ValueError(f"TableModalProcessor expects type='table', got {block.get('type')!r}")

        body = block.get("table_body") or []
        body_repr = json.dumps(body, sort_keys=True) if isinstance(body, (list, dict)) else str(body)

        img_path = block.get("img_path") or ""
        image_b64 = ""
        if img_path and Path(img_path).exists():
            image_b64 = _encode_image_to_base64(img_path)

        cache_key = self._cache_key(body_repr, image_b64)
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["block_id"] = block_id
            cached["from_cache"] = True
            return cached

        prompt = TABLE_PROMPT.format(
            entity_name=f"table_{block_id}",
            table_img_path=img_path or "(no image)",
            table_caption=block.get("table_caption") or "None",
            table_body=body_repr or "(empty)",
            table_footnote=block.get("table_footnote") or "None",
        )

        try:
            raw = self.vision_func(prompt, image_b64 or None, TABLE_ANALYSIS_SYSTEM)
        except Exception as exc:
            log.warning("vision call failed for table block %s: %s", block_id, exc)
            return self._fail(block_id, str(exc), body_repr)

        result = self._parse(raw, block_id, body_repr)
        self._cache[cache_key] = result
        self._persist(cache_key, result)
        return result

    def _parse(self, raw: str, block_id: str, body_repr: str) -> Dict[str, Any]:
        try:
            data = _robust_json_parse(raw)
            table_md = (data.get("table_markdown") or "").strip()
            caption = (data.get("caption") or "").strip()
            entity = data.get("entity_info") or {}
            if not caption:
                caption = entity.get("summary") or ""
            if not table_md:
                raise ValueError("table_markdown missing or empty in response")
            return {
                "block_id": block_id,
                "table_markdown": table_md,
                "caption": caption,
                "description": "",  # legacy field; not used in merge anymore
                "table_body": body_repr,
                "entities": [entity] if entity else [],
                "raw_response": raw,
                "confidence": 1.0,
            }
        except ValueError as exc:
            log.info("json parse failed for table block %s: %s", block_id, exc)
            fallback = _strip_thinking_tags(raw)
            return {
                "block_id": block_id,
                "table_markdown": "",
                "caption": fallback[:200],
                "description": fallback,
                "table_body": body_repr,
                "entities": [],
                "raw_response": raw,
                "confidence": 0.5,
            }

    @staticmethod
    def _fail(block_id: str, reason: str, body_repr: str) -> Dict[str, Any]:
        return {
            "block_id": block_id,
            "description": "",
            "caption": f"[table processing failed: {reason}]",
            "table_body": body_repr,
            "entities": [],
            "raw_response": "",
            "confidence": 0.0,
        }


def process_blocks(
    blocks: List[Dict[str, Any]],
    image_processor: Optional[ImageModalProcessor] = None,
    table_processor: Optional[TableModalProcessor] = None,
    block_id_fn: Optional[Callable[[Dict[str, Any], int], str]] = None,
    budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run modal processors over a block list.

    Returns the original list with each image/table block augmented with
    modal_result. Raises RuntimeError when budget is reached (caller decides
    whether to bump and retry — see V6B_MODAL_CALL_BUDGET in the plan).
    """
    calls = 0
    out: List[Dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        new_block = dict(block)
        bid = block_id_fn(block, idx) if block_id_fn else f"blk_{idx}"
        btype = block.get("type")
        if btype == "image" and image_processor is not None:
            if budget is not None and calls >= budget:
                raise RuntimeError(f"modal call budget exceeded at {calls}; pass V6B_MODAL_CALL_BUDGET=N to override")
            new_block["modal_result"] = image_processor.process(block, bid)
            if not new_block["modal_result"].get("from_cache"):
                calls += 1
        elif btype == "table" and table_processor is not None:
            if budget is not None and calls >= budget:
                raise RuntimeError(f"modal call budget exceeded at {calls}; pass V6B_MODAL_CALL_BUDGET=N to override")
            new_block["modal_result"] = table_processor.process(block, bid)
            if not new_block["modal_result"].get("from_cache"):
                calls += 1
        out.append(new_block)
    return out
