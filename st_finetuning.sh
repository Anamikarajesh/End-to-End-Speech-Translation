#!/usr/bin/env bash
# Short list of commands that completed successfully for Stage 1 prep/training.
# Run individually; training command is long-running.

conda run -n aml python -m fairseq_cli.train \
  preprocessing/data/iwslt_databin \
  --save-dir training/st_stage2_finetune \
  --task speech_to_text \
  --train-subset train \
  --valid-subset valid \
  --max-source-positions 30000 \
  --load-pretrained-encoder-from training/asr_stage1/checkpoint_39_6000.pt \
  --arch s2t_conformer \
  --encoder-layers 16 \
  --decoder-layers 6 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --lr 0.002 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 1000 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --dropout 0.1 \
  --max-tokens 10000 \
  --update-freq 32 \
  --max-update 2250 \
  --patience 10 \
  --log-interval 10 \
  --save-interval-updates 500 \
  --keep-best-checkpoints 5 \
  --find-unused-parameters \
  --tensorboard-logdir log/tensorboard/st_stage2
