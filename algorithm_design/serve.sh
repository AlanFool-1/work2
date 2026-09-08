#!/usr/bin/env bash
set -euo pipefail

WEB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_PORT="${1:-8765}"
MKDOCS_BIN="${MKDOCS_BIN:-/root/anaconda3/envs/torch/bin/mkdocs}"

exec "$MKDOCS_BIN" serve \
  --config-file "$WEB_ROOT/algorithm_design/mkdocs.yml" \
  --dev-addr "127.0.0.1:$WEB_PORT"
