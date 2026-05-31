"""Vendored slices of RAG-Anything (HKUDS) — see VENDOR_NOTES.md."""

from tamubot.vendor.raganything.batch_parser import batch_parse
from tamubot.vendor.raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    process_blocks,
)
from tamubot.vendor.raganything.parser import (
    MineruParser,
    Parser,
    get_parser,
    list_parsers,
    register_parser,
)
from tamubot.vendor.raganything.resilience import retry_call

__all__ = [
    "Parser",
    "MineruParser",
    "ImageModalProcessor",
    "TableModalProcessor",
    "process_blocks",
    "batch_parse",
    "retry_call",
    "register_parser",
    "get_parser",
    "list_parsers",
]
