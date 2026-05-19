#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
DATASET="${DATASET:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_20260518a_closebridge/alignment_tc_student_vnext_dataset_20260518a_closebridge.npz}"
INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260518a_edgepair_label_only/alignment_tc_student_vnext_best.pt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260518a_closebridge}"

mkdir -p "$OUT_ROOT"
exec "$PYTHON_BIN" "$ROOT/scripts/train_alignment_tc_student_vnext.py" \
  --dataset "$DATASET" \
  --output_dir "$OUT_ROOT" \
  --stage stage3_student_finetune \
  --init_ckpt "$INIT_CKPT" \
  --epochs 18 \
  --batch_size 64 \
  --lr 8e-5 \
  --horizon 8 \
  --max_pos_step 0.0015 \
  --max_yaw_step 0.0060 \
  --phase1_target_axis_weights 1.25,1.25,1.35,0.25,0.25,0.80 \
  --phase2_target_axis_weights 1.0,1.50,1.0,0.25,0.25,1.75 \
  --phase1_action_axis_weights 1.25,1.25,1.35,0.75 \
  --phase2_action_axis_weights 1.0,1.50,1.0,1.75 \
  --phase1_yaw_dir_weight 0.90 \
  --phase2_yaw_dir_weight 1.00 \
  --phase1_verified_weight_boost 1.10 \
  --enable_close_ready_bridge_supervision \
  --phase1_close_ready_loss_weight 1.00 \
  --phase1_handoff_ready_loss_weight 0.75 \
  --phase1_bridge_xy_boost 0.75 \
  --phase1_bridge_z_boost 0.85 \
  --phase1_bridge_yaw_floor 0.25 \
  --phase1_bridge_yaw_cap 1.25
