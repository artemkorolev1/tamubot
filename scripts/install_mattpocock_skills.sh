#!/usr/bin/env bash
set -euo pipefail
REPO="mattpocock/skills"
BRANCH="main"
SKILLS=(tdd diagnose zoom-out improve-codebase-architecture grill-with-docs to-prd)
TARGETS=("$HOME/.claude/skills" "/workspace/.claude/skills")

TREE=$(curl -fsSL "https://api.github.com/repos/$REPO/git/trees/$BRANCH?recursive=1")
for skill in "${SKILLS[@]}"; do
  src="skills/engineering/$skill/"
  paths=$(echo "$TREE" | python3 -c "import sys,json;d=json.load(sys.stdin);[print(t['path']) for t in d['tree'] if t['type']=='blob' and t['path'].startswith('$src')]")
  if [[ -z "$paths" ]]; then echo "ERROR: no files for $skill" >&2; exit 1; fi
  for tgt in "${TARGETS[@]}"; do
    for p in $paths; do
      rel="${p#"$src"}"
      dest="$tgt/$skill/$rel"
      mkdir -p "$(dirname "$dest")"
      curl -fsSL "https://raw.githubusercontent.com/$REPO/$BRANCH/$p" -o "$dest"
    done
  done
  echo "installed $skill"
done
