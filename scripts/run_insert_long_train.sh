#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Long training: insert_onto_square_peg (50 000 steps, 1× GPU)
# Defaults can be overridden with env vars for faster machine-specific bring-up.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

export WANDB_MODE=offline
export WANDB_CONSOLE=off
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export HF_HOME="${HF_HOME:-/home/guoning/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/insert_long_train}"
RUN_ID_NOTE="${RUN_ID_NOTE:-insert_depth_force_50k}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
MAX_STEPS="${MAX_STEPS:-50000}"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate vla-adapter

echo "=== Starting insert_onto_square_peg long training (50k steps) ==="
echo "PYTHONPATH=${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "RUN_ROOT_DIR=${RUN_ROOT_DIR}"
echo "RUN_ID_NOTE=${RUN_ID_NOTE}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "MAX_STEPS=${MAX_STEPS}"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --config_file_path pretrained_models/configs \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --use_minivlm True \
  --rlbench_data_root data/rlbench_data \
  --rlbench_task_name insert_onto_square_peg \
  --use_l1_regression True \
  --use_proprio True \
  --use_depth True \
  --use_force True \
  --use_pro_version True \
  --use_lora True \
  --lora_rank 64 \
  --num_images_in_input 2 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --max_steps "${MAX_STEPS}" \
  --save_freq "${SAVE_FREQ}" \
  --num_steps_before_decay 40000 \
  --lr_warmup_steps 500 \
  --image_aug True \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --wandb_project vla_insert_long \
  --wandb_log_freq 10 \
  --phase Training

echo "=== Training complete ==="
