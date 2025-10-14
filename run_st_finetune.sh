#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] Activate the 'ml' virtual environment before launching training." >&2
  exit 1
fi

cd "$PROJECT_ROOT"

fairseq-hydra-train \
  --config-dir configs/fairseq \
  --config-name st_finetune \
  "$@"
