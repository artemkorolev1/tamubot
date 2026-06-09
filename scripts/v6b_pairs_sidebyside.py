#!/usr/bin/env python3
"""Emit a single self-contained side-by-side viewer for an iteration's pairs.

For an iter dir produced by scripts/v6b_build_comparison_pairs.py, this reads every
original/<NNN_stem>.md and processed/<NNN_stem>.md and writes one HTML file with two
synced-scroll panes (ORIGINAL | PROCESSED), a stem dropdown, and an optional PDF-pages
pane. No server, no network: open it directly via file:// .

  docker exec tamubot-dev-1 python scripts/v6b_pairs_sidebyside.py \
    data/syllabi/_preprocessing_lab/_review

Writes <iter_dir>/pairs_sidebyside.html .
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    iter_dir = Path(sys.argv[1])
    orig_dir = iter_dir / "original"
    proc_dir = iter_dir / "processed"
    if not orig_dir.is_dir() or not proc_dir.is_dir():
        print(f"error: {iter_dir} has no original/ + processed/ — build pairs first")
        return 1

    pairs: list[dict] = []
    for omd in sorted(orig_dir.glob("*.md")):
        name = omd.name  # NNN_<stem>.md
        pmd = proc_dir / name
        if not pmd.exists():
            continue
        stem = name.split("_", 1)[1].rsplit(".md", 1)[0]
        seq = name.split("_", 1)[0]
        # pdf page list (relative paths so file:// works from iter_dir)
        page_dir = iter_dir / "pdf_pages" / stem
        pages = (
            [f"pdf_pages/{stem}/{p.name}" for p in sorted(page_dir.glob("*.png"))]
            if page_dir.is_dir()
            else []
        )
        pairs.append(
            {
                "seq": seq,
                "stem": stem,
                "original": omd.read_text(encoding="utf-8"),
                "processed": pmd.read_text(encoding="utf-8"),
                "pages": pages,
            }
        )

    if not pairs:
        print("error: no aligned original/processed pairs found")
        return 1

    data_json = json.dumps(pairs, ensure_ascii=False)
    html = _TEMPLATE.replace("/*__DATA__*/", data_json).replace(
        "__TITLE__", f"side-by-side · {iter_dir.name} · {len(pairs)} pairs"
    )
    out = iter_dir / "pairs_sidebyside.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(pairs)} pairs)")
    print("open via file:// — on Windows the container path maps to C:\\dev\\TAMUBOT\\...")
    return 0


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 :root{--maroon:#500000}
 *{box-sizing:border-box}
 body{font:13px system-ui,sans-serif;margin:0;background:#faf9f7;color:#222}
 header{background:var(--maroon);color:#fff;padding:8px 14px;display:flex;gap:14px;
   align-items:center;position:sticky;top:0;z-index:9;flex-wrap:wrap}
 header b{font-size:14px}
 select{font:13px system-ui;padding:3px 6px}
 label{font-size:12px;display:flex;gap:5px;align-items:center;cursor:pointer}
 .wrap{display:grid;grid-template-columns:1fr 1fr;gap:0;height:calc(100vh - 42px)}
 .wrap.pdf{grid-template-columns:1fr 1fr 420px}
 .col{overflow:auto;height:100%;border-right:1px solid #ddd}
 .colhead{position:sticky;top:0;background:#f0ede9;color:var(--maroon);font-weight:600;
   padding:5px 12px;border-bottom:1px solid #ddd;z-index:2}
 pre{white-space:pre-wrap;word-break:break-word;margin:0;padding:10px 14px;
   font:12px/1.5 ui-monospace,Consolas,monospace}
 .body{flex:1;overflow:auto}
 .body img{width:100%;border:1px solid #ddd;margin:0 0 8px}
 .col{display:flex;flex-direction:column}
 mark{background:#fff2a8;padding:0 1px}
 .hint{font-size:11px;color:#ffd7d7}
</style></head><body>
<header>
 <b>side-by-side</b>
 <select id="sel"></select>
 <select id="layout">
  <option value="processed,pdf">PROCESSED | SOURCE PDF</option>
  <option value="original,pdf">ORIGINAL | SOURCE PDF</option>
  <option value="original,processed">ORIGINAL | PROCESSED</option>
  <option value="processed,original">PROCESSED | ORIGINAL</option>
 </select>
 <span id="anchor" class="hint"></span>
 <label><input type="checkbox" id="sync" checked> sync scroll</label>
 <span class="hint">ORIGINAL = faithful bronze · PROCESSED = RAG-visible (boilerplate/duplicate removed) · PDF = ground truth</span>
</header>
<div class="wrap" id="wrap">
 <div class="col" id="cL"><div class="colhead" id="hL"></div><div class="body" id="bL"></div></div>
 <div class="col" id="cR"><div class="colhead" id="hR"></div><div class="body" id="bR"></div></div>
</div>
<script>
const DATA = /*__DATA__*/;
const sel=document.getElementById('sel'), layout=document.getElementById('layout'),
 bL=document.getElementById('bL'), bR=document.getElementById('bR'),
 hL=document.getElementById('hL'), hR=document.getElementById('hR'),
 anchor=document.getElementById('anchor'), sync=document.getElementById('sync');
DATA.forEach((d,i)=>{const o=document.createElement('option');o.value=i;
 o.textContent=`${d.seq} · ${d.stem}`;sel.appendChild(o);});
const LABEL={original:'ORIGINAL (faithful extracted text)',
 processed:'PROCESSED (what RAG sees)', pdf:'SOURCE PDF (ground truth)'};
function fillPane(head,body,kind,d){
 head.textContent=LABEL[kind];
 if(kind==='pdf'){body.innerHTML=d.pages.map(p=>`<img loading="lazy" src="${p}">`).join('');}
 else{body.innerHTML='';const pre=document.createElement('pre');pre.textContent=d[kind];body.appendChild(pre);}
 body.scrollTop=0;}
function render(){const d=DATA[+sel.value];const [l,r]=layout.value.split(',');
 fillPane(hL,bL,l,d); fillPane(hR,bR,r,d);
 anchor.textContent=`orig ${d.original.length} ch · proc ${d.processed.length} ch · ${d.pages.length} pages`;}
sel.addEventListener('change',render);
layout.addEventListener('change',render);
// proportional synced scroll between the two visible panes
let lock=false;
function link(a,b){a.addEventListener('scroll',()=>{if(!sync.checked||lock)return;lock=true;
 const r=a.scrollTop/Math.max(1,a.scrollHeight-a.clientHeight);
 b.scrollTop=r*(b.scrollHeight-b.clientHeight);lock=false;});}
link(bL,bR);link(bR,bL);
render();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
