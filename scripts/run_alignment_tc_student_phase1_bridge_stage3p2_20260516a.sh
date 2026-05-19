#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
DATASET="${DATASET:-runtime_artifacts/alignment_tc_diffusion/student_phase1_bridge_dataset_20260516a/alignment_tc_student_phase1_bridge_dataset_20260516a.npz}"
INIT_CKPT="${INIT_CKPT:-runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260515n/alignment_tc_student_vnext_best.pt}"
OUT_DIR="${OUT_DIR:-runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_phase1_bridge_stage3p2_20260516a}"

mkdir -p "$OUT_DIR"

env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" scripts/train_alignment_tc_student_vnext.py \
  --dataset "$DATASET" \
  --output_dir "$OUT_DIR" \
  --stage stage3_student_finetune \
  --init_ckpt "$INIT_CKPT" \
  --epochs 24 \
  --batch_size 64 \
  --lr 8e-5 \
  --horizon 8 \
  --max_pos_step 0.0015 \
  --max_yaw_step 0.0060 \
  --phase1_target_axis_weights 1.0,1.35,1.0,0.25,0.25,1.35 \
  --phase2_target_axis_weights 1.0,1.50,1.0,0.25,0.25,1.75 \
  --phase1_action_axis_weights 1.0,1.40,1.0,1.45 \
  --phase2_action_axis_weights 1.0,1.50,1.0,1.75 \
  --phase1_yaw_dir_weight 1.10 \
  --phase2_yaw_dir_weight 1.00 \
  --phase1_verified_weight_boost 1.05 \
  --enable_phase1_bridge_repair_losses \
  --phase1_sign_y_weight 0.35 \
  --phase1_sign_yaw_weight 0.50 \
  --phase1_mag_floor_y_weight 0.15 \
  --phase1_mag_floor_yaw_weight 0.20
