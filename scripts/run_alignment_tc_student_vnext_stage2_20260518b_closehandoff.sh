#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"
DATASET="${DATASET:-$ROOT/runtime_artifacts/alignment_tc_diffusion/student_vnext_dataset_20260518a_closebridge/alignment_tc_student_vnext_dataset_20260518a_closebridge.npz}"
INIT_CKPT="${INIT_CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260518a_closebridge/alignment_tc_student_vnext_best.pt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage2_20260518b_closehandoff}"

mkdir -p "$OUT_ROOT"
exec "$PYTHON_BIN" "$ROOT/scripts/train_alignment_tc_student_vnext.py" \
  --dataset "$DATASET" \
  --output_dir "$OUT_ROOT" \
  --stage stage2_teacher_forcing \
  --init_ckpt "$INIT_CKPT" \
  --epochs 10 \
  --batch_size 64 \
  --lr 8e-5 \
  --horizon 8 \
  --max_pos_step 0.0015 \
  --max_yaw_step 0.0060 \
  --enable_close_ready_bridge_supervision \
  --phase1_close_ready_loss_weight 1.25 \
  --phase1_handoff_ready_loss_weight 1.00 \
  --phase1_bridge_xy_boost 0.50 \
  --phase1_bridge_z_boost 0.75 \
  --phase1_bridge_yaw_floor 0.20 \
  --phase1_bridge_yaw_cap 1.00 \
  --freeze_all_but_close_ready_handoff_heads
