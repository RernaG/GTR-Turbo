#!/bin/bash
set -euo pipefail

MODEL_PATH="/data/czd/model/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="/data/czd/GTR-Turbo/Turbo_P24/checkpoints_3b"

mkdir -p "$OUTPUT_DIR"

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
    --config_file config_zero2_single.yaml --main_process_port 29380 ../main.py \
    --env-name gym_cards/Points24-v0 \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 10 \
    --eval-num-per-episode 50 \
    --num-env-steps 4096 \
    --num-steps 256 \
    --grad-accum-steps 32 \
    --max-new-tokens 384 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path "$MODEL_PATH" \
    --use-lora \
    --train-vision all \
    --output_dir "$OUTPUT_DIR" \
    --tht-guide SFT \
    --tag TurboSFT-3B
