export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
export PYTHONPATH=/home/user/anamika_ml_project/fairseq && \
conda run -n aml python -m fairseq_cli.train preprocessing/data/iwslt_databin \
    --save-dir training/st_stage2_finetune_v7 \
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
    --lr-scheduler inverse_sqrt \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 \
    --lr 0.002 \
    --warmup-updates 1000 \
    --encoder-freezing-updates 1000 \
    --max-tokens 10000 \
    --update-freq 32 \
    --max-update 2250 \
    --patience 25 \
    --log-interval 10 \
    --save-interval-updates 500 \
    --keep-best-checkpoints 5 \
    --find-unused-parameters \
    --tensorboard-logdir log/tensorboard/st_stage2_v5 \
    | tee log/training/stage2_v5_train.log