#!/usr/bin/env bash
# Prepare speech-to-text assets (SentencePiece + transcript exports).
#
# The speech pipeline relies on manifest TSVs with `audio`, `n_frames`, and
# `text` columns. This helper script copies the tokenizer assets into the data
# directory and optionally exports raw transcripts for tokenizer training.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $(basename "$0") /path/to/spm_6k.model /path/to/spm_6k.txt" >&2
  exit 1
fi

SPM_MODEL_SRC=$1
SPM_DICT_SRC=$2

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR="${REPO_ROOT}/data"
MANIFEST_TRAIN="${DATA_DIR}/combined_train_raw.tsv"
MANIFEST_VALID="${DATA_DIR}/combined_valid_raw.tsv"
TEXT_OUTPUT_DIR="${DATA_DIR}"

mkdir -p "${DATA_DIR}"

SPM_MODEL_BASENAME=$(basename "${SPM_MODEL_SRC}")
SPM_DICT_BASENAME=$(basename "${SPM_DICT_SRC}")

cp "${SPM_MODEL_SRC}" "${DATA_DIR}/${SPM_MODEL_BASENAME}"
cp "${SPM_DICT_SRC}" "${DATA_DIR}/${SPM_DICT_BASENAME}"

# Optional: export transcripts for tokenizer sanity checks / additional training
python "${REPO_ROOT}/scripts/export_text_and_vocab.py" \
  --train-manifest "${MANIFEST_TRAIN}" \
  --valid-manifest "${MANIFEST_VALID}" \
  --output-dir "${TEXT_OUTPUT_DIR}"
