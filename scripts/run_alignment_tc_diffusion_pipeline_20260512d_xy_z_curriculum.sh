#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_DIR="${RAW_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/privileged_teacher_raw_20260512d_xy_z_curriculum_10ep}"
TEACHER_DIR="${TEACHER_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/teacher_20260512d_xy_z_curriculum}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_20260512d_xy_z_curriculum}"
SHADOW_DIR="${SHADOW_DIR:-$ROOT/runtime_artifacts/alignment_tc_diffusion/shadow_3ep_20260512d_xy_z_curriculum}"

RAW_INPUT_NPZ="${RAW_INPUT_NPZ:-$RAW_DIR/alignment_tc_privileged_teacher_raw_20260512d_xy_z_curriculum_10ep.npz}"
RAW_REPORT_JSON="${RAW_REPORT_JSON:-$RAW_DIR/alignment_tc_privileged_teacher_raw_report_20260512d_xy_z_curriculum_10ep.json}"
RAW_AUDIT_JSON="${RAW_AUDIT_JSON:-$RAW_DIR/alignment_tc_privileged_teacher_raw_audit_20260512d_xy_z_curriculum_10ep.json}"
TEACHER_NPZ="${TEACHER_NPZ:-$TEACHER_DIR/alignment_tc_diffusion_teacher_20260512d_xy_z_curriculum.npz}"
TEACHER_REPORT="${TEACHER_REPORT:-$TEACHER_DIR/alignment_tc_diffusion_teacher_report_20260512d_xy_z_curriculum.json}"

RAW_DIR="$RAW_DIR" \
OUTPUT_DIR="$RAW_DIR" \
TAG="20260512d_xy_z_curriculum_10ep" \
CHECKPOINT_DIR="/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt" \
bash "$ROOT/scripts/run_alignment_tc_privileged_teacher_collect_10ep_20260512d_xy_z_curriculum.sh"

INPUT_NPZ="$RAW_INPUT_NPZ" \
OUTPUT_NPZ="$TEACHER_NPZ" \
REPORT_JSON="$TEACHER_REPORT" \
STAGE_BUCKETS="near_contact_refine,micro_contact_refine" \
MIN_ROWS="32" \
MAX_POS_STEP="0.0015" \
MAX_YAW_STEP="0.0060" \
bash "$ROOT/scripts/run_alignment_tc_diffusion_build_teacher.sh"

DATASET="$TEACHER_NPZ" \
OUTPUT_DIR="$TRAIN_DIR" \
EPOCHS="60" \
BATCH_SIZE="64" \
LR="3e-4" \
DELTA_WEIGHT="1.0" \
HEATMAP_WEIGHT="0.25" \
CONFIDENCE_WEIGHT="0.35" \
TRAJECTORY_WEIGHT="1.5" \
PROGRESS_WEIGHT="0.55" \
RISK_WEIGHT="0.45" \
STOP_WEIGHT="0.25" \
SMOOTH_WEIGHT="0.05" \
DELTA_XY_WEIGHT="4.0" \
DELTA_Z_WEIGHT="2.0" \
DELTA_YAW_WEIGHT="1.5" \
DELTA_ROLLPITCH_WEIGHT="0.5" \
TRAJECTORY_XY_WEIGHT="4.0" \
TRAJECTORY_Z_WEIGHT="2.0" \
TRAJECTORY_YAW_WEIGHT="1.5" \
PROGRESS_XY_WEIGHT="2.0" \
PROGRESS_Z_WEIGHT="1.5" \
PROGRESS_YAW_WEIGHT="1.0" \
bash "$ROOT/scripts/run_alignment_tc_diffusion_train.sh"

ALIGNMENT_TC_DIFFUSION_CKPT="$TRAIN_DIR/alignment_tc_diffusion_refiner_best.pt" \
OUTPUT_DIR="$SHADOW_DIR" \
CHECKPOINT_DIR="/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0001+lora-r64+dropout-0.0--image_aug--insert_vision_only_current_phasebucket_weighted_lr1e4_guard_50k_20260427a--30000_chkpt" \
bash "$ROOT/scripts/run_alignment_tc_diffusion_shadow_3ep.sh"
