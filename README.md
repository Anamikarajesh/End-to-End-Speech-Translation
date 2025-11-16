# End-to-End Speech Translation (Marathi → Hindi)

This project reproduces the full Marathi→Hindi speech translation pipeline—data acquisition, preprocessing, Fairseq-based pretraining/fine-tuning, Hugging Face experiments, inference, and evaluation scripts. The repository keeps only code and lightweight configs; large data/model artifacts are downloaded on demand using the scripts described below.

---
## Table of Contents
1. [Repository Layout](#repository-layout)
2. [External Assets](#external-assets)
   - [1. IWSLT 2023 Marathi–Hindi Dataset](#1-iwslt-2023-marathi–hindi-dataset)
   - [2. SentencePiece Tokenizer (6k vocab)](#2-sentencepiece-tokenizer-6k-vocab)
   - [3. ASR Stage‑1 (Pretraining) Checkpoint](#3-asr-stage-1-pretraining-checkpoint)
   - [4. Stage‑2 Fine-Tuning Checkpoints & BLEU Logs](#4-stage-2-fine-tuning-checkpoints--bleu-logs)
3. [Quick Start](#quick-start)
4. [Environment Setup](#environment-setup)
   - [Fairseq pipeline (`aml` env)](#fairseq-pipeline-aml-env)
5. [Data Preparation](#data-preparation)
6. [Training Workflows](#training-workflows)
   - [ASR Stage‑1 (Encoder warm-up)](#asr-stage-1-encoder-warm-up)
   - [Stage‑2 NMT Fine-Tuning (Fairseq)](#stage-2-nmt-fine-tuning-fairseq)
7. [Evaluation and Inference](#evaluation-and-inference)
8. [TensorBoard Visualization](#tensorboard-visualization)
9. [Troubleshooting & Tips](#troubleshooting--tips)
10. [Citation & Licensing](#citation--licensing)
11. [Acknowledgements](#acknowledgements)

---
## Repository Layout

```
anamika_ml_project/
├── dataset/                     # IWSLT audio + manifests (external download)
├── fairseq/                     # Fairseq source (recommend git submodule)
├── inference/                   # Translation & BLEU evaluation scripts
│   ├── translate_audio.py
│   └── evaluate_bleu.py
├── log/
│   └── tensorboard/             # TensorBoard event files (auto-generated)
├── preprocessing/
│   ├── scripts/                 # Data preparation utilities
│   └── data/                    # TSV manifests, SPM tokenizer files
├── tools/                       # Utility helpers
├── training/                    # Checkpoints, generation outputs (external)
├── .gitignore
└── ReadMe.md                    # Project documentation
```

Large directories (`dataset/`, `training/`, `output/`, `log/`, `alternative/hf_cache/`, etc.) are ignored via `.gitignore`. Fetch artifacts with the helper scripts referenced below.

---
## External Assets

| Asset | Size | Location | Notes |
|-------|------|----------|-------|
| IWSLT2023 Mr→Hi raw audio | ~21 GB | [official IWSLT site](https://iwslt.org/2023) / self-hosted mirror | 16 kHz wav + transcripts
| SentencePiece tokenizer (6k) | <5 MB | Included / rebuild script | `spm_6k.model` + vocab
| ASR Stage‑1 checkpoint | ~1.2 GB | Hosted artifact (see script) | Warm-start encoder
| ST Stage‑2 checkpoints | ~5 GB | Hosted artifact | Decoder warm-up + full fine-tune
| Hugging Face models | ~6 GB | Hugging Face Hub cache | Download automatically (`wav2vec2`, `mt5`)

### 1. IWSLT 2023 Marathi–Hindi Dataset
Use `scripts/download_dataset.sh` to mirror your storage (replace URLs with accessible sources):

```bash
#!/usr/bin/env bash
set -euo pipefail
TARGET="dataset/iwslt2023_mr-hi"
mkdir -p "${TARGET}"

# Replace the following URLs with your own mirrors
wget -O "${TARGET}/train.tar.gz" https://storage.example.com/iwslt2023_mr-hi_train.tar.gz
wget -O "${TARGET}/dev.tar.gz"   https://storage.example.com/iwslt2023_mr-hi_dev.tar.gz
wget -O "${TARGET}/test.tar.gz"  https://storage.example.com/iwslt2023_mr-hi_test.tar.gz

for split in train dev test; do
  tar -xzf "${TARGET}/${split}.tar.gz" -C "${TARGET}"
  rm -f "${TARGET}/${split}.tar.gz"
done
```

Expected layout:

```
dataset/iwslt2023_mr-hi/
├── train/wav/*.wav
├── dev/wav/*.wav
└── test/wav/*.wav
```

### 2. SentencePiece Tokenizer (6k vocab)
Tokenizer assets live under `preprocessing/data/tokenizer/`. Rebuild if needed:

```bash
python preprocessing/scripts/train_sentencepiece.py \
  --input preprocessing/data/iwslt_train_text.txt \
  --model_prefix spm_6k \
  --vocab_size 6000
```

Outputs:
- `spm_6k.model`
- `spm_6k.vocab`
- `spm_6k.txt` (optional plain-text export)

### 3. ASR Stage‑1 (Pretraining) Checkpoint
Download the encoder warm-start checkpoint into `training/asr_stage1/`:

```bash
mkdir -p training/asr_stage1
curl -L -o training/asr_stage1/checkpoint_39_6000.pt \
  https://storage.example.com/checkpoint_39_6000.pt
```

### 4. Stage‑2 Fine-Tuning Checkpoints & BLEU Logs
Two-phase runs are packaged separately:
- Decoder warm-up (`training/st_stage2_finetune_v6a_decoder_only/`)
- Full fine-tune (`training/st_stage2_finetune_v6b_full_finetune/`)

Host them in a release or cloud bucket, then mirror using `scripts/download_checkpoints.sh` (placeholder):

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p training

wget -O training/st_stage2_v6a.tar.gz https://storage.example.com/st_stage2_v6a.tar.gz
wget -O training/st_stage2_v6b.tar.gz https://storage.example.com/st_stage2_v6b.tar.gz

for pkg in st_stage2_v6a st_stage2_v6b; do
  tar -xzf "training/${pkg}.tar.gz" -C training
  rm -f "training/${pkg}.tar.gz"
done

---
## Quick Start

```bash
# Clone (with submodules if using Fairseq as a submodule)
git clone --recursive git@github.com:Anamikarajesh/End-to-End-Speech-Translation.git
cd End-to-End-Speech-Translation

# Download required assets (edit scripts for your storage bucket)
bash scripts/download_dataset.sh
bash scripts/download_checkpoints.sh

# Prepare Fairseq environment & preprocess manifests
conda env create -f envs/aml.yml      # optional helper file
conda activate aml
python preprocessing/scripts/prepare_s2t_databin.py \
  --train-tsv preprocessing/data/iwslt_train_raw.tsv \
  --valid-tsv preprocessing/data/iwslt_valid_raw.tsv \
  --dataset-root dataset/iwslt2023_mr-hi \
  --output-dir preprocessing/data/iwslt_databin \
  --spm-model preprocessing/data/tokenizer/spm_6k.model

# Stage-2 fine-tuning (Fairseq)
bash training/st_finetune_v6b.sh

# BLEU evaluation
python inference/evaluate_bleu.py \
  --generate-file training/st_stage2_finetune_v6b_full_finetune/generate-test.txt \
  --output-json training/st_stage2_finetune_v6b_full_finetune/final_bleu_score/bleu.json


# Single-file inference
audio_path=dataset/iwslt2023_mr-hi/test/wav/sample.wav
python inference/translate_audio.py \
  --config configs/inference.yaml \
  --checkpoint training/st_stage2_finetune_v6b_full_finetune/checkpoint_best.pt \
  --audio-path "$audio_path" \
  --output-text outputs/sample_translation.txt
```

---
## Environment Setup

### Fairseq pipeline (`aml` env)

```bash
conda create -n aml python=3.9 -y
conda activate aml

pip install --upgrade pip
pip install -r fairseq/requirements.txt
pip install -e fairseq

pip install tensorboard sacrebleu sentencepiece sox soundfile
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```
## Data Preparation

`preprocessing/scripts/prepare_s2t_databin.py` consumes TSV manifests and builds Fairseq bins:

```bash
python preprocessing/scripts/prepare_s2t_databin.py \
  --train-tsv preprocessing/data/iwslt_train_raw.tsv \
  --valid-tsv preprocessing/data/iwslt_valid_raw.tsv \
  --dataset-root dataset/iwslt2023_mr-hi \
  --output-dir preprocessing/data/iwslt_databin \
  --spm-model preprocessing/data/tokenizer/spm_6k.model
```

Manifest columns required: `id`, `audio`, `n_frames`, `tgt_text`. Ensure `audio` points to actual wav files (absolute paths recommended).

Optional helpers:
- `preprocessing/scripts/create_char_dict.py`
- `tools/sox_resample.sh` for normalizing sample rates

---
## Training Workflows

### ASR Stage‑1 (Encoder warm-up)

Use `training/asr_pretraining.sh` (not tracked here) to obtain `training/asr_stage1/checkpoint_39_6000.pt`. This step depends on your ASR corpus; skip if you already have the checkpoint.

### Stage‑2 NMT Fine-Tuning (Fairseq)

Two-phase approach consistently improved BLEU.

#### Phase 1: Decoder warm-up (v6a)

```bash
bash training/st_finetune_v6a.sh
```

Key flags (see script):
- `--task speech_to_text`
- `--load-pretrained-encoder-from training/asr_stage1/checkpoint_39_6000.pt`
- Encoder frozen (`--encoder-freezing-updates 0`)
- `--max-update 750`, `--patience 10`

Outputs: `training/st_stage2_finetune_v6a_decoder_only/`

#### Phase 2: Full fine-tune (v6b)

```bash
bash training/st_finetune_v6b.sh
```

Highlights:
- Restores from v6a checkpoint
- Thaws encoder after 1000 updates
- `--update-freq 32`, `--max-tokens 10000`, `--max-update 2250`
- `--eval-bleu --eval-bleu-args '{"beam": 5}'`
- Saves best checkpoints by BLEU

Outputs: `training/st_stage2_finetune_v6b_full_finetune/`

Logs and events stream to `log/tensorboard/st_stage2_v6b/`.



Outputs:
```
alternative/hf_finetune_v1/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-final/
└── logs/
```

---
## Evaluation and Inference

### BLEU Scoring

```bash
python inference/evaluate_bleu.py \
  --generate-file training/st_stage2_finetune_v6b_full_finetune/generate-test.txt \
  --output-json training/st_stage2_finetune_v6b_full_finetune/final_bleu_score/bleu.json \
  --spm-model preprocessing/data/tokenizer/spm_6k.model \
  --ref-file dataset/iwslt2023_mr-hi/test/text/hi.txt \
  --beam 5
```

Produces BLEU scores plus per-sentence outputs in JSON.

### Single Audio Translation

```bash
python inference/translate_audio.py \
  --config configs/inference.yaml \
  --checkpoint training/st_stage2_finetune_v6b_full_finetune/checkpoint_best.pt \
  --audio-path dataset/iwslt2023_mr-hi/test/wav/sample.wav \
  --output-text outputs/sample_translation.txt
```

Ensure `configs/inference.yaml` references the correct `spm_model` and `data_bin` directories.

---
## TensorBoard Visualization

```bash
conda activate newml
tensorboard --logdir log/tensorboard --host 0.0.0.0 --port 6006
```

Major runs:
- `st_stage2_v2/`
- `st_stage2_v6a/`
- `st_stage2_v6b/`

Open `http://localhost:6006` to view training dynamics.

---
## Troubleshooting & Tips

- **SSH push failures**: Ensure your GitHub SSH key is added (`ssh -T git@github.com`).
- **CUDA OOM / driver resets**: Lower `--batch_size`, `--update-freq`; disable `--fp16`; consider CPU-only trials.
- **Audio load errors**: Multi-channel wavs are averaged to mono in `hf_finetune.py`; adjust if needed.
- **Tokenizer mismatch**: Keep `spm_6k.model/.vocab` synchronized between preprocessing and inference.
- **Large asset hosting**: Use Git LFS sparingly or host elsewhere (S3, Hugging Face Hub, GDrive). Document download steps clearly.

---
## Citation & Licensing

- **Dataset**: Cite IWSLT 2023 shared task publications.
- **Fairseq**: MIT License (see `fairseq/LICENSE`).
- **This project**: My preferred `LICENSE` (MIT)

---
## Acknowledgements

- IWSLT 2023 organizers for the Marathi–Hindi corpus.
- Fairseq community for the speech-to-text toolkit.
- Hugging Face team for open-source models and infrastructure.
- Contributors and collaborators who tested or reviewed this pipeline.

Questions or issues? Open an issue on GitHub or reach out directly. Happy translating!
