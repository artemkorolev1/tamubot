"""HTTP client for a vLLM-served NuExtract3 (OpenAI-compatible).

Mirrors NuExtractExtractor's interface but offloads generation to the sidecar,
which does continuous batching server-side — so extract_text_batch just fans out
concurrent requests rather than padding locally. Greedy (temperature 0) keeps
output equivalent to the in-process path.
"""

from __future__ import annotations

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx

from tamubot.ingestion.clients.nuextract_client import SYLLABUS_TEMPLATE, parse_extract
from tamubot.rag.models_v4 import SyllabusExtract

if TYPE_CHECKING:
    from PIL.Image import Image

_TEMPLATE_JSON = json.dumps(SYLLABUS_TEMPLATE, indent=4)


def build_chat_payload(content: list[dict], *, model: str, max_tokens: int = 1024) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"template": _TEMPLATE_JSON, "enable_thinking": False},
    }


def parse_chat_response(data: dict[str, Any]) -> SyllabusExtract:
    return parse_extract(data["choices"][0]["message"]["content"])


class NuExtractHTTPClient:
    """OpenAI-compatible client for the NuExtract3 vLLM sidecar. Interface-compatible
    with NuExtractExtractor (extract_text / extract_text_batch / extract_image)."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "numind/NuExtract3",
        timeout: float = 180.0,
        max_concurrency: int = 8,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._max_concurrency = max_concurrency
        self._client = httpx.Client(timeout=timeout)

    def _post(self, content: list[dict], *, max_new_tokens: int = 1024) -> SyllabusExtract:
        r = self._client.post(
            f"{self.base_url}/chat/completions",
            json=build_chat_payload(content, model=self.model, max_tokens=max_new_tokens),
        )
        r.raise_for_status()
        return parse_chat_response(r.json())

    def extract_text(self, markdown: str) -> SyllabusExtract:
        return self._post([{"type": "text", "text": markdown}])

    def extract_text_batch(self, markdowns: list[str]) -> list[SyllabusExtract]:
        # vLLM continuous-batches server-side; just fan out bounded-concurrency requests.
        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            return list(pool.map(self.extract_text, markdowns))

    def extract_image(self, image: "Image") -> SyllabusExtract:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return self._post([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}])
