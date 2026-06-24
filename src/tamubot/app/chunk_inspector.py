"""Side-by-side chunk inspector: raw PDF (left) vs v6b semantic chunks (right),
with each chunk's text highlighted on the page it came from.

Run (from repo root):
    streamlit run src/tamubot/app/chunk_inspector.py --server.headless true

Pure read-only over already-processed artifacts — no pipeline run, no LLM calls.
Geometry comes from each block's persisted Docling bbox; for files parsed before
bbox persistence it falls back to a PyMuPDF text search, so it works on the
existing corpus immediately.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import streamlit as st

# Make ``tamubot`` importable when run via ``streamlit run`` from the repo root.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tamubot.ingestion.pipeline_v6b import paths  # noqa: E402
from tamubot.ingestion.pipeline_v6b.chunk_provenance import (  # noqa: E402
    align_chunks_to_blocks,
    resolve_annotations,
)

DATA_ROOT = "data/syllabi"
SELECTED_COLOR = "#e63946"  # red — the chunk under inspection

st.set_page_config(layout="wide", page_title="v6b Chunk Inspector")


# ── Data loading (cached on file mtime so edits/re-runs refresh) ──────────────


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def discover_stems() -> list[str]:
    files = glob.glob(f"{DATA_ROOT}/*/v6b/silver/02_chunk/*.json")
    return sorted(Path(f).name[: -len(".json")] for f in files)


@st.cache_data(show_spinner=False)
def load_stem(stem: str, _sig: tuple) -> dict:
    """Load blocks + chunks for a stem. Prefers the tagged chunks (03_tag) so
    boilerplate/duplicate flags show; falls back to the base chunks (02_chunk).
    ``_sig`` (file mtimes) busts the cache when artifacts change."""
    blocks = json.loads(paths.bronze_blocks_path(stem).read_text(encoding="utf-8"))

    tag_path = paths.silver_tag_path(stem)
    chunk_source = "03_tag"
    if tag_path.exists():
        chunk_doc = json.loads(tag_path.read_text(encoding="utf-8"))
    else:
        chunk_source = "02_chunk"
        chunk_doc = json.loads(paths.silver_chunk_semantic_path(stem).read_text(encoding="utf-8"))
    chunks = chunk_doc["chunks"]

    aligned = align_chunks_to_blocks(blocks, chunks)
    pages_per_chunk = [sorted({b.get("page_idx") for b in a if b.get("page_idx")}) for a in aligned]
    return {
        "blocks": blocks,
        "chunks": chunks,
        "aligned": aligned,
        "pages_per_chunk": pages_per_chunk,
        "chunk_source": chunk_source,
        "n_blocks": len(blocks),
        "n_matched": sum(len(a) for a in aligned),
    }


def _stem_sig(stem: str) -> tuple:
    return (
        _mtime(paths.bronze_blocks_path(stem)),
        _mtime(paths.silver_tag_path(stem)),
        _mtime(paths.silver_chunk_semantic_path(stem)),
    )


# ── Sidebar: pick a file ──────────────────────────────────────────────────────

stems = discover_stems()
if not stems:
    st.error(f"No processed chunks found under {DATA_ROOT}/*/v6b/silver/02_chunk/. Run the pipeline first.")
    st.stop()

with st.sidebar:
    st.title("Chunk Inspector")
    stem = st.selectbox("Syllabus", stems, key="stem")
    st.caption(f"{len(stems)} processed file(s)")

data = load_stem(stem, _stem_sig(stem))
pdf_path = paths.raw_path(stem)
chunks = data["chunks"]
aligned = data["aligned"]

with st.sidebar:
    st.divider()
    st.metric("Chunks", len(chunks))
    st.metric("Blocks aligned", f"{data['n_matched']}/{data['n_blocks']}")
    st.caption(f"chunk source: `{data['chunk_source']}`")
    if not pdf_path.exists():
        st.warning(f"Raw PDF missing: {pdf_path}")

# Track which chunk is selected for highlighting.
if st.session_state.get("_stem_for_sel") != stem:
    st.session_state["sel_chunk"] = 0
    st.session_state["_stem_for_sel"] = stem
selected = st.session_state.get("sel_chunk", 0)

# ── Two-pane layout ───────────────────────────────────────────────────────────

left, right = st.columns([3, 2], gap="medium")

with right:
    st.subheader("Chunks")
    st.caption("Click **Highlight** to locate a chunk on the PDF →")
    for i, c in enumerate(chunks):
        pages = data["pages_per_chunk"][i]
        tags = []
        if c.get("is_boilerplate"):
            tags.append("🟡 boilerplate")
        if c.get("is_duplicate"):
            tags.append("🔴 duplicate")
        for f in c.get("flags") or []:
            tags.append(f"`{f}`")
        is_sel = i == selected
        border = "border-left:4px solid #e63946;" if is_sel else "border-left:4px solid #ccc;"
        hp = c.get("header_path") or "—"
        with st.container():
            st.markdown(
                f"<div style='{border}padding:4px 10px;margin-bottom:2px;'>"
                f"<b>#{i}</b> · {hp} "
                f"<span style='color:#888'>· {c.get('token_count', '?')} tok"
                f"{' · p' + ','.join(map(str, pages)) if pages else ''}</span><br>"
                f"{' '.join(tags)}</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns([1, 5])
            if cols[0].button("Highlight", key=f"hl_{i}", type="primary" if is_sel else "secondary"):
                st.session_state["sel_chunk"] = i
                st.rerun()
            with cols[1].expander("text"):
                st.text(c.get("content", ""))

with left:
    st.subheader(f"#{selected} · {chunks[selected].get('header_path') or '—'}")
    if not pdf_path.exists():
        st.info("PDF unavailable — cannot render.")
    else:
        blocks_for_chunk = aligned[selected]
        annotations = resolve_annotations(pdf_path, blocks_for_chunk, color=SELECTED_COLOR)
        pages = data["pages_per_chunk"][selected]
        if not annotations:
            st.warning(
                "No boxes located for this chunk (table-only or pre-bbox parse with no text match). Showing full PDF."
            )
        try:
            from streamlit_pdf_viewer import pdf_viewer
        except ImportError:
            st.error("streamlit-pdf-viewer not installed. `pip install streamlit-pdf-viewer`")
            st.stop()
        pdf_viewer(
            str(pdf_path),
            annotations=annotations,
            scroll_to_page=pages[0] if pages else 1,
            width=750,
            key=f"pdf_{stem}_{selected}",
        )
