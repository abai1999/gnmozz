#!/bin/bash
# Planner-only baseline training for VLA2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/home/guoning/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

RUN_ID_NOTE="${RUN_ID_NOTE:-vla2_planner_baseline_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$ROOT/outputs/planner_baseline}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
MAX_STEPS="${MAX_STEPS:-50000}"
TRAIN_SEED="${TRAIN_SEED:-}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME:-vla-adapter}"

CMD=(torchrun --standalone --nnodes 1 --nproc-per-node 1 "$ROOT/vla-scripts/finetune.py" \
  --config_file_path "$ROOT/pretrained_models/configs" \
  --vlm_path "$ROOT/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" \
  --use_minivlm True \
  --rlbench_data_root "$ROOT/data/rlbench_data" \
  --rlbench_task_name "${TASK_NAME:-insert_onto_square_peg}" \
  --use_l1_regression True \
  --use_proprio True \
  --use_depth False \
  --use_force False \
  --use_pro_version True \
  --use_lora True \
  --lora_rank 64 \
  --num_images_in_input 2 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-2e-4}" \
  --max_steps "${MAX_STEPS}" \
  --save_freq "${SAVE_FREQ}" \
  --num_steps_before_decay 40000 \
  --lr_warmup_steps 500 \
  --image_aug True \
  --run_root_dir "$RUN_ROOT_DIR" \
  --run_id_note "$RUN_ID_NOTE" \
  --wandb_project "${WANDB_PROJECT:-vla2_planner_baseline}" \
  --wandb_log_freq 10 \
  --phase Training)

if [[ -n "$TRAIN_SEED" ]]; then
  CMD+=(--seed "$TRAIN_SEED")
fi

echo "[vla2] training planner baseline"
echo "[vla2] ROOT=$ROOT"
echo "[vla2] RUN_ROOT_DIR=$RUN_ROOT_DIR"
echo "[vla2] TASK_NAME=${TASK_NAME:-insert_onto_square_peg}"
"${CMD[@]}"
