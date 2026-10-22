#!/usr/bin/env bash
# Rebuild only this app's generated browser bundle; never delete an arbitrary path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export UV_LINK_MODE=copy
uv run marimo export html-wasm app.py -o build --mode run -f
test -f build/index.html
echo "Ready: serve build/ with python -m http.server 8000 --directory build"
