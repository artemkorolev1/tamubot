#!/bin/sh
# Recover HF cache symlinks from Cygwin XSym text stubs (Windows host wrote them
# as files instead of symlinks). Each stub holds the target path on line 4.
# Idempotent: skips files that already aren't XSym pointers.
set -u
ROOT="${1:-/home/claude/.cache/huggingface/hub}"

fixed=0
skipped=0
total=0
# Recurse — TableFormer's tm_config.json lives at
# snapshots/<sha>/model_artifacts/tableformer/<accurate|fast>/
find "$ROOT"/models--*/snapshots -type f 2>/dev/null | while IFS= read -r f; do
    total=$((total + 1))
    head_bytes=$(head -c 4 "$f" 2>/dev/null)
    if [ "$head_bytes" != "XSym" ]; then
        skipped=$((skipped + 1))
        echo "S $f"  # progress line; tally summed at end
        continue
    fi
    target=$(sed -n '4p' "$f")
    if [ -z "$target" ]; then
        echo "WARN no target in $f" >&2
        continue
    fi
    rm -f "$f"
    ln -s "$target" "$f"
    echo "F $f -> $target"
done | awk '
    /^F / { fixed++ }
    /^S / { skipped++ }
    END { printf "fixed=%d skipped=%d\n", fixed, skipped }
'
