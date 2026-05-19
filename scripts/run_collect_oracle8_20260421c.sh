#!/usr/bin/env bash
set -euo pipefail

cd /home/guoning/code/VLA

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_CACHE_ROOT=/mnt/ssd/guoning/hf-cache
export HF_HOME=/mnt/ssd/guoning/hf-cache
export HF_HUB_CACHE=/mnt/ssd/guoning/hf-cache/hub
export HUGGINGFACE_HUB_CACHE=/mnt/ssd/guoning/hf-cache/hub
export TRANSFORMERS_CACHE=/mnt/ssd/guoning/hf-cache/transformers
export TORCH_HOME=/mnt/ssd/guoning/hf-cache/torch
export TIMM_HOME=/mnt/ssd/guoning/hf-cache/timm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_LOCAL_FILES_ONLY=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_support_resync_oracle_full8_20260421c

xvfb-run -a /home/guoning/my_conda_envs/vla-adapter/bin/python scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir /home/guoning/code/VLA/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt \
  --task_name insert_onto_square_peg \
  --modes oracle_target_upper_bound \
  --num_episodes 8 \
  --max_steps 300 \
  --episode_indices 32,33,34,35,36,37,38,39 \
  --output_root /home/guoning/code/VLA/eval_logs/insert_onto_square_peg/support_rows_resync_20260421c \
  --name_suffix seed3407_full_oracle8 \
  --alignment_ckpt /home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt \
  --target_provider_ckpt /home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_target_delta_proxy_20260419c/target_delta_predictor_best.pt \
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
  --record_teacher_truth_metrics \
  --support_states_output_npz /home/guoning/code/VLA/runtime_artifacts/residual_data/insert_phase1_support_resync_oracle_full8_20260421c/support_states.npz
