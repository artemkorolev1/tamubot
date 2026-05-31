"""Parser abstraction adapted from RAG-Anything (HKUDS).

Block format is preserved verbatim from upstream so block lists produced here
remain interoperable with any tool that consumes RAG-Anything's native JSON.
LightRAG coupling and unused parser backends are not vendored.

See VENDOR_NOTES.md in this directory for the upstream commit SHA and the
specific changes from the source files.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)


class Parser:
    """Base class for document parsers.

    Subclasses implement parse_pdf / parse_image / parse_document and return a
    list of content blocks in the format:

        {"type": "text",     "text": str, "page_idx": int}
        {"type": "image",    "img_path": str, "image_caption": str, "image_footnote": str, "page_idx": int}
        {"type": "table",    "img_path": str, "table_caption": str, "table_footnote": str, "table_body": list, "page_idx": int}
        {"type": "equation", "img_path": str, "text": str, "text_format": str, "page_idx": int}
        {"type": "heading",  "text": str, "level": int, "page_idx": int}   # tamubot extension

    The "heading" type is a tamubot-specific addition that upstream does not
    emit; downstream chunkers use it to build header_path. It does not break
    upstream consumers that ignore unknown types.
    """

    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    TEXT_FORMATS = {".txt", ".md"}

    @staticmethod
    def _is_url(path: str) -> bool:
        try:
            r = urllib.parse.urlparse(str(path))
            return all([r.scheme, r.netloc])
        except ValueError:
            return False

    @staticmethod
    def block_id(stem: str, page_idx: int, bbox: Optional[tuple] = None, idx: int = 0) -> str:
        """Stable per-block identifier.

        sha256(stem + page_idx + bbox + idx)[:16]. Falls back to (stem, page,
        idx) when bbox is not available (heading items, etc.). Stable across
        re-runs of the same input.
        """
        key = f"{stem}|{page_idx}|{bbox or ''}|{idx}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def check_installation(self) -> bool:
        raise NotImplementedError


class MineruParser(Parser):
    """Placeholder for MinerU backend, wired in Phase 2 of the v6b plan.

    The Phase 2 implementation calls a MinerU sidecar over HTTP (see plan
    section "Phase 2 — MinerU sidecar + hybrid routing"). The body is left
    blank intentionally: importing the class is fine, instantiating and
    calling parse_pdf in Phase 0 / 1 raises NotImplementedError so misuse is
    surfaced loudly rather than silently routing to Docling.
    """

    def check_installation(self) -> bool:
        return False

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "MineruParser is a Phase 2 placeholder; sidecar not yet stood up. See plans/toasty-weaving-taco.md."
        )

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        return self.parse_pdf(file_path, output_dir=output_dir, method=method, lang=lang, **kwargs)


_REGISTRY: Dict[str, type] = {}


def register_parser(name: str, parser_class: type) -> None:
    """Register a parser implementation under a short name (e.g. "docling")."""
    if not issubclass(parser_class, Parser):
        raise TypeError(f"{parser_class!r} must subclass Parser")
    _REGISTRY[name.lower()] = parser_class


def get_parser(name: str) -> Parser:
    """Look up a registered parser by name. Raises KeyError if not found."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"No parser registered for {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[key]()


def list_parsers() -> Dict[str, str]:
    """Return {name: class_qualname} for all registered parsers."""
    return {name: cls.__qualname__ for name, cls in _REGISTRY.items()}


register_parser("mineru", MineruParser)
