#!/usr/bin/env sh
# Force the Inspect View log viewer into dark (night) mode, including the top toolbar/
# navbar (which is a CSS-module component that ignores Bootstrap's data-bs-theme).
#
# Two edits to the packaged viewer's index.html:
#   1) data-bs-theme="dark" on <html>  (themes all Bootstrap surfaces)
#   2) an injected <style> override that darkens the top bar via substring class
#      selectors ([class*="navbar..."]) which survive the build's hash suffixes.
#
# Edits site-packages, so it does NOT survive a container rebuild / `pip install inspect_ai`
# upgrade — re-run it then (idempotent). Needs write perms on site-packages:
#
#     docker exec -u root tamubot-dev-1 sh scripts/inspect_view_dark.sh         # dark
#     docker exec -u root tamubot-dev-1 sh scripts/inspect_view_dark.sh light   # revert
set -e
MODE="${1:-dark}"
python - "$MODE" <<'PY'
import os, sys, inspect_ai
mode = sys.argv[1]
html = os.path.join(os.path.dirname(inspect_ai.__file__), "_view", "dist", "index.html")
src = open(html, encoding="utf-8").read()

# 1) theme attribute
import re
src = re.sub(r'data-bs-theme="[a-z]+"', f'data-bs-theme="{mode}"', src)

MARK = "preproc-dark-override"
OVERRIDE = """<style id="preproc-dark-override">
[data-bs-theme="dark"] [class*="navbarWrapper"],
[data-bs-theme="dark"] [class*="navbarConfig"],
[data-bs-theme="dark"] [class*="navbarContainer"],
[data-bs-theme="dark"] [class*="sidebarHeader"],
[data-bs-theme="dark"] [class*="workspacePath"],
[data-bs-theme="dark"] .navbar {
  background: var(--bs-body-bg) !important;
  color: var(--bs-body-color) !important;
  border-color: var(--bs-border-color) !important;
}
[data-bs-theme="dark"] [class*="navbar"] .text-muted,
[data-bs-theme="dark"] [class*="navbar"] a { color: var(--bs-body-color) !important; }
</style>"""

if mode == "dark" and MARK not in src:
    src = src.replace("</head>", OVERRIDE + "\n</head>", 1)
elif mode != "dark" and MARK in src:
    src = re.sub(r'<style id="preproc-dark-override">.*?</style>\n?', "", src, flags=re.S)

open(html, "w", encoding="utf-8").write(src)
print(f"set data-bs-theme=\"{mode}\"" + (" + header override" if mode == "dark" else " (override removed)") + f" in {html}")
PY
