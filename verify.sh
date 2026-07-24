#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src/deepstack:${ROOT_DIR}/src/tilesight${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m ae.cli verify "$@"
