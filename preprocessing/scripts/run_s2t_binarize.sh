#!/usr/bin/env bash

# This script runs Fairseq's speech-to-text preprocessing. It reads the
# combined TSV manifests, extracts fbank80 features, applies SpecAugment
# statistics, tokenizes transcripts with SentencePiece, and writes the
# resulting data-bin in one shot.

set -euo pipefail

# --- 1. Set up paths (override via environment variables as needed) ---
ASR_DATA_RAW_ROOT=${ASR_DATA_RAW_ROOT:-"/home/user/anamika_ml_project/preprocessing/data"}
SENTENCEPIECE_MODEL=${SENTENCEPIECE_MODEL:-"/home/user/anamika_ml_project/preprocessing/data/spm_6k.model"}
CONFIG_YAML=${CONFIG_YAML:-"/home/user/anamika_ml_project/preprocessing/data/config_asr.yaml"}
ASR_DATABIN_ROOT=${ASR_DATABIN_ROOT:-"/home/user/anamika_ml_project/preprocessing/data/asr_combined_databin"}
WORKERS=${WORKERS:-8}

FAIRSEQ_ROOT="/home/user/anamika_ml_project/fairseq"
PYTHONPATH=${PYTHONPATH:-${FAIRSEQ_ROOT}}

TRAIN_MANIFEST="${ASR_DATA_RAW_ROOT}/combined_train_raw.tsv"
VALID_MANIFEST="${ASR_DATA_RAW_ROOT}/combined_valid_raw.tsv"

mkdir -p "${ASR_DATABIN_ROOT}"

echo "--- Starting FAIRSEQ S2T Preprocessing ---"

echo "Config YAML      : ${CONFIG_YAML}"
echo "Train manifest   : ${TRAIN_MANIFEST}"
echo "Valid manifest   : ${VALID_MANIFEST}"
echo "SPM model        : ${SENTENCEPIECE_MODEL}"
echo "Output root      : ${ASR_DATABIN_ROOT}"
echo "Workers          : ${WORKERS}"

TEXT_CONFIG=$(SENTENCEPIECE_MODEL="${SENTENCEPIECE_MODEL}" python - <<'PY'
import json
import os
cfg = {
    "_target_": "fairseq.data.text_compressor.TextCompressorConfig",
    "text_tokenizer": "sentencepiece",
    "spm_model": os.path.abspath(os.environ["SENTENCEPIECE_MODEL"]),
}
print(json.dumps(cfg))
PY
)

PYTHONPATH="${PYTHONPATH}" python -m fairseq_cli.preprocess \
  --config-yaml "${CONFIG_YAML}" \
  --task speech_to_text \
  --train-manifest "${TRAIN_MANIFEST}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --output-root "${ASR_DATABIN_ROOT}" \
  --text-config "${TEXT_CONFIG}" \
  --workers "${WORKERS}"

echo "--- Preprocessing complete. ---"
echo "Stage-1 ASR data-bin ready at: ${ASR_DATABIN_ROOT}"
