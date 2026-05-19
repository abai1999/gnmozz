#!/bin/bash
# -------------------------------------------------------------------
# Paper-faithful planner-core control run for RLBench insert.
# Keeps the current RLBench data/eval path, but swaps only the planner
# core to the original VLA-Adapter-style action head.
# -------------------------------------------------------------------
set -euo pipefail

export WANDB_MODE=offline
export WANDB_CONSOLE=off
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export HF_HOME="${HF_HOME:-/home/guoning/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
RUN_ID_NOTE="${RUN_ID_NOTE:-insert_vision_only_paper_faithful_50k_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/insert_long_train}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
MAX_STEPS="${MAX_STEPS:-50000}"
TRAIN_SEED="${TRAIN_SEED:-}"

eval "$(conda shell.bash hook)"
conda activate vla-adapter

echo "=== Starting paper-faithful planner-core control run ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "RUN_ID_NOTE=${RUN_ID_NOTE}"
echo "RUN_ROOT_DIR=${RUN_ROOT_DIR}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "TRAIN_SEED=${TRAIN_SEED:-<unset>}"

CMD=(torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --config_file_path pretrained_models/configs \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --use_minivlm True \
  --rlbench_data_root data/rlbench_data \
  --rlbench_task_name insert_onto_square_peg \
  --use_l1_regression True \
  --use_proprio True \
  --use_depth False \
  --use_force False \
  --use_pro_version True \
  --planner_core_variant paper_faithful \
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
  --phase Training)

if [[ -n "${TRAIN_SEED}" ]]; then
  CMD+=(--seed "${TRAIN_SEED}")
fi

"${CMD[@]}"

echo "=== Paper-faithful control training complete ==="
