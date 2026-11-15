#!/usr/bin/env bash
# Rebuild the data from the latest results, then serve the site at http://localhost:${PORT:-8000}
set -euo pipefail
cd "$(dirname "$0")"
python3 build.py
echo "Open http://localhost:${PORT:-8000}  (Ctrl+C to stop)"
python3 -m http.server "${PORT:-8000}"
