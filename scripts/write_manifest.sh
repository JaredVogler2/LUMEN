#!/usr/bin/env bash
# Write manifest.json for Posit Connect (git-backed publishing or `rsconnect deploy manifest`).
#
#   pip install rsconnect-python        # a publishing tool; not a runtime dependency of LUMEN
#   scripts/write_manifest.sh [path/to/python]
#
# The interpreter decides the Python and pip versions recorded in the manifest; use one
# that matches a Python installed on the Connect server and that has pip (an
# environment made by uv has none).  Re-run after any change to the files that are
# deployed: the manifest lists each file with a checksum, and a test compares them.
# Exclusion patterns may contain "**" only once (a limit of the tool).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-$(command -v python3)}"
rsconnect write-manifest fastapi \
  --overwrite \
  --entrypoint app:app \
  --python "$PY" \
  --requirements-file requirements.txt \
  -x "tests/**" -x "docs/**" -x "prompt_series/**" -x "scripts/**" \
  -x "requirements-dev.txt" -x "pyproject.toml" -x "README.md" -x ".gitignore" \
  -x ".venv*/**" -x ".pytest_cache/**" -x "**/__pycache__/*" -x "**/*.pyc" -x ".lumen-data/**" -x "lumen-data/**" -x "manifest.json" \
  .
echo "manifest.json written: $(python3 -c 'import json;m=json.load(open("manifest.json"));print(m["metadata"]["appmode"], m["metadata"]["entrypoint"], "python", m["python"]["version"], len(m["files"]), "files")')"
