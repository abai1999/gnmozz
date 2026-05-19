#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/guoning/code/VLA"
cd "$ROOT"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(bash "$ROOT/scripts/choose_idle_gpu.sh")"
  export CUDA_VISIBLE_DEVICES
fi

PY="/home/guoning/my_conda_envs/vla-adapter/bin/python"
BASELINE_SCORER="/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt"
TARGET_PROVIDER="/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt"
CHECKPOINT_DIR="/home/guoning/code/VLA/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt"

NEAR_READY_DIR="/home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_near_ready_candidate_20260420v"
TRAIN_DIR="/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_residual_adapter_20260420v"
EVAL_ROOT="/home/guoning/code/VLA/eval_logs/insert_onto_square_peg/no_priv_student_runtime_20260420v_residual_adapter"
mkdir -p "$NEAR_READY_DIR" "$TRAIN_DIR" "$EVAL_ROOT"

"$PY" scripts/build_near_ready_candidate_dataset.py \
  --input_npz /home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_teacher_support_states_demo_grasp_yaw_20260419a/support_states.npz \
  --input_npz /home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthready_positive_support_20260420i/support_states.npz \
  --input_npz /home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_truthready_anchor_support_20260420j/support_states.npz \
  --output_path "$NEAR_READY_DIR/near_ready_candidate_dataset.npz" \
  --near_xy_max_mult 4.0 \
  --near_abs_z_max_mult 2.0 \
  --near_yaw_max_mult 2.0 \
  --exclude_already_ready \
  --keep_ready_fraction 0.25

"$PY" scripts/audit_near_ready_reachability.py \
  --dataset_npz "$NEAR_READY_DIR/near_ready_candidate_dataset.npz" \
  --baseline_ckpt "$BASELINE_SCORER" \
  --batch_size 64 | tee "$TRAIN_DIR/reachability_audit.stdout.json"

env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PY" -u scripts/train_near_ready_residual_adapter.py \
  --dataset_npz "$NEAR_READY_DIR/near_ready_candidate_dataset.npz" \
  --baseline_ckpt "$BASELINE_SCORER" \
  --output_dir "$TRAIN_DIR" \
  --epochs 12 \
  --batch_size 64 \
  --lr 3e-4 \
  --rank_margin 0.35 \
  --clip_rho 0.35 \
  --lambda_ce 1.0 \
  --lambda_pair 1.0 \
  --lambda_margin 1.0 \
  --lambda_l2 0.02

env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
HF_CACHE_ROOT=/mnt/ssd/guoning/hf-cache HF_HOME=/mnt/ssd/guoning/hf-cache HF_HUB_CACHE=/mnt/ssd/guoning/hf-cache/hub \
HUGGINGFACE_HUB_CACHE=/mnt/ssd/guoning/hf-cache/hub TRANSFORMERS_CACHE=/mnt/ssd/guoning/hf-cache/transformers \
TORCH_HOME=/mnt/ssd/guoning/hf-cache/torch TIMM_HOME=/mnt/ssd/guoning/hf-cache/timm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 HF_LOCAL_FILES_ONLY=1 HF_HUB_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
xvfb-run -a "$PY" scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task_name insert_onto_square_peg \
  --modes learned_target_mainline \
  --num_episodes 3 \
  --max_steps 300 \
  --episode_indices 0,1,2 \
  --output_root "$EVAL_ROOT" \
  --name_suffix seed3407_residual_adapter_smoke_mp4 \
  --alignment_ckpt "$BASELINE_SCORER" \
  --residual_score_adapter_ckpt "$TRAIN_DIR/near_ready_residual_adapter_best.pt" \
  --target_provider_ckpt "$TARGET_PROVIDER" \
  --planner_no_depth \
  --planner_no_force \
  --enable_alignment_close_veto \
  --close_veto_xy_threshold 0.006 \
  --close_veto_abs_z_threshold 0.003 \
  --close_veto_ready_streak_frames 1 \
  --close_veto_settle_steps 0 \
  --learned_residual_scale 0.50 \
  --max_residual_pos 0.006 \
  --max_alignment_corrections_per_window 120 \
  --outer_rescue_min_xy 0.10 \
  --outer_rescue_min_abs_z 0.30 \
  --eval_seed 3407 \
  --close_latch_enabled \
  --close_latch_steps 32 \
  --use_legacy_teacher_candidate_bank_for_scorer \
  --disable_alignment_physical_mask \
  --enforce_no_privileged_runtime
