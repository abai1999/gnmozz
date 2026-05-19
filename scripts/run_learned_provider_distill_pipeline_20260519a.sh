#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/guoning/my_conda_envs/vla-adapter/bin/python}"

INPUT_NPZ_PHASE1="${INPUT_NPZ_PHASE1:-$ROOT/runtime_artifacts/alignment_tc_diffusion/alignment_tc_planner_state_expert_recovery_raw_20260514n_zonly80_fixed_merged_raw.npz}"
INPUT_NPZ_PHASE2="${INPUT_NPZ_PHASE2:-$ROOT/runtime_artifacts/alignment_tc_diffusion/planner_phase2_insert_teacher_20260517a_edgepair/alignment_tc_planner_phase2_insert_teacher_20260517a_edgepair.npz}"
DISTILL_OUT_ROOT="${DISTILL_OUT_ROOT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/learned_provider_distill_20260519a}"
TARGET_OUT_ROOT="${TARGET_OUT_ROOT:-$DISTILL_OUT_ROOT/train_target_delta_predictor}"
HANDOFF_OUT_ROOT="${HANDOFF_OUT_ROOT:-$DISTILL_OUT_ROOT/train_handoff_predictor}"
SMOKE_OUT_ROOT="${SMOKE_OUT_ROOT:-$DISTILL_OUT_ROOT/smoke_learned_provider_3ep}"

TASK_NAME="${TASK_NAME:-insert_onto_square_peg}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/ssd/guoning/VLA_runtime/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vision_only_paper_faithful_phasebucket_weighted_50k_20260426b--50000_chkpt}"
STUDENT_CKPT="${STUDENT_CKPT:-$ROOT/runtime_artifacts/alignment_tc_diffusion/train_alignment_tc_student_vnext_stage3_20260518a_closebridge/alignment_tc_student_vnext_best.pt}"

mkdir -p "$DISTILL_OUT_ROOT" "$TARGET_OUT_ROOT" "$HANDOFF_OUT_ROOT" "$SMOKE_OUT_ROOT"

DISTILL_NPZ="$DISTILL_OUT_ROOT/learned_provider_distill_20260519a.npz"
DISTILL_REPORT="$DISTILL_OUT_ROOT/learned_provider_distill_report_20260519a.json"

echo "[1/4] Building learned-provider distillation dataset"
"$PYTHON_BIN" "$ROOT/scripts/build_learned_provider_distill_dataset.py" \
  --input_npz "$INPUT_NPZ_PHASE1" \
  --input_npz "$INPUT_NPZ_PHASE2" \
  --source_name "phase1_teacher_20260514n_zonly80_fixed" \
  --source_name "phase2_insert_teacher_20260517a_edgepair" \
  --output_npz "$DISTILL_NPZ" \
  --report_json "$DISTILL_REPORT" \
  --min_band_label 0

echo "[2/4] Training target-delta predictor"
"$PYTHON_BIN" "$ROOT/scripts/train_target_delta_predictor.py" \
  --dataset_npz "$DISTILL_NPZ" \
  --output_dir "$TARGET_OUT_ROOT" \
  --epochs 12 \
  --batch_size 64 \
  --lr 2e-4 \
  --val_ratio 0.1 \
  --seed 3407

echo "[3/4] Training handoff predictor"
"$PYTHON_BIN" "$ROOT/scripts/train_handoff_predictor.py" \
  --dataset_npz "$DISTILL_NPZ" \
  --output_dir "$HANDOFF_OUT_ROOT" \
  --epochs 12 \
  --batch_size 64 \
  --lr 2e-4 \
  --val_ratio 0.1 \
  --seed 3407

echo "[4/4] Running 3-episode learned-provider smoke"
exec env xvfb-run -a "$PYTHON_BIN" "$ROOT/scripts/evaluate_rlbench.py" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name "$TASK_NAME" \
  --alignment_tc_student_vnext_ckpt "$STUDENT_CKPT" \
  --use_stage_aware_refiner \
  --stage_refiner_mode full \
  --enable_alignment_tc_student_vnext_shadow \
  --enable_alignment_tc_student_vnext_apply \
  --target_provider_mode learned \
  --target_provider_ckpt "$TARGET_OUT_ROOT/target_delta_predictor_best.pt" \
  --handoff_provider_ckpt "$HANDOFF_OUT_ROOT/handoff_predictor_best.pt" \
  --alignment_tc_student_vnext_collector_like \
  --enable_alignment_tc_student_vnext_ready_gate \
  --alignment_tc_student_vnext_close_ready_threshold 0.5 \
  --alignment_tc_student_vnext_handoff_ready_threshold 0.5 \
  --collector_like_demo_reset \
  --enforce_no_privileged_runtime \
  --num_episodes 3 \
  --episode_indices "5,8,19" \
  --max_steps 340 \
  --eval_seed 3407 \
  --output_dir "$SMOKE_OUT_ROOT" \
  --record_video \
  --write_episode_videos \
  --no_best_gif
