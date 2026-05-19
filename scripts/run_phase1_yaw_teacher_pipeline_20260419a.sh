#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/guoning/code/VLA"
RUN_TAG="20260419a"
TASK_NAME="insert_onto_square_peg"
SEED=3407
COLLECT_EPISODES=40
BENCH_EPISODES=10
MAX_STEPS=300
GPU_ID="${CUDA_VISIBLE_DEVICES:-1}"

CHECKPOINT_DIR="${REPO_ROOT}/outputs/insert_long_train/configs+insert_onto_square_peg+b4+lr-0.0002+lora-r64+dropout-0.0--image_aug--insert_vo_layernorm_s3407--40000_chkpt"
# The original 0418h bootstrap scorer artifact is missing in this workspace.
# Use the closest existing canonical scorer lineage to restart teacher collection
# and keep the collection / scorer training path reproducible.
BOOTSTRAP_ALIGNMENT_CKPT="${BOOTSTRAP_ALIGNMENT_CKPT:-/home/guoning/code/VLA/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_20260419k_oracle_target_twolayer_step_student/pose_field_scorer_best_pose.pt}"

COLLECT_ROOT="${REPO_ROOT}/eval_logs/${TASK_NAME}/phase1_demo_grasp_yaw_teacher_${RUN_TAG}"
COLLECT_SUFFIX="seed${SEED}_collect_oracle_exec_yaw_formal${COLLECT_EPISODES}_mp4"
SUPPORT_DIR="${REPO_ROOT}/runtime_artifacts/residual_data/insert_phase1_teacher_support_states_demo_grasp_yaw_${RUN_TAG}"
SUPPORT_NPZ="${SUPPORT_DIR}/support_states.npz"

DATASET_DIR="${REPO_ROOT}/runtime_artifacts/residual_data/insert_phase1_visual_posefield_candidates_demo_grasp_yaw_${RUN_TAG}"
DATASET_NPZ="${DATASET_DIR}/candidates.npz"
SCORER_DIR="${REPO_ROOT}/runtime_artifacts/stage_refiner/insert_phase1_posefield_demo_grasp_yaw_${RUN_TAG}_visual_student"
STUDENT_CKPT="${SCORER_DIR}/pose_field_scorer_best_pose.pt"

BENCH_ROOT="${REPO_ROOT}/eval_logs/${TASK_NAME}/phase1_demo_grasp_yaw_student_benchmark_${RUN_TAG}"
LOG_DIR="${REPO_ROOT}/runtime_artifacts/logs"
LOG_FILE="${LOG_DIR}/phase1_yaw_teacher_pipeline_${RUN_TAG}.log"

mkdir -p "${SUPPORT_DIR}" "${DATASET_DIR}" "${SCORER_DIR}" "${BENCH_ROOT}" "${LOG_DIR}"

export WANDB_MODE=offline
export WANDB_CONSOLE=off
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export COPPELIASIM_ROOT="/home/guoning/CoppeliaSim"
export QT_QPA_PLATFORM=xcb
export QT_QPA_PLATFORM_PLUGIN_PATH="/home/guoning/CoppeliaSim"
export QT_PLUGIN_PATH="/home/guoning/CoppeliaSim"
export LD_LIBRARY_PATH="/home/guoning/CoppeliaSim:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO_ROOT}"

echo "[pipeline] RUN_TAG=${RUN_TAG}" | tee "${LOG_FILE}"
echo "[pipeline] collect_root=${COLLECT_ROOT}" | tee -a "${LOG_FILE}"
echo "[pipeline] support_npz=${SUPPORT_NPZ}" | tee -a "${LOG_FILE}"
echo "[pipeline] dataset_npz=${DATASET_NPZ}" | tee -a "${LOG_FILE}"
echo "[pipeline] scorer_dir=${SCORER_DIR}" | tee -a "${LOG_FILE}"
echo "[pipeline] bench_root=${BENCH_ROOT}" | tee -a "${LOG_FILE}"

echo "[pipeline] 1/4 collect privileged teacher rows (${COLLECT_EPISODES} episodes, full ${MAX_STEPS} steps)" | tee -a "${LOG_FILE}"
xvfb-run -a /home/guoning/my_conda_envs/vla-adapter/bin/python scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --task_name "${TASK_NAME}" \
  --modes oracle_executed_pregrasp_collect \
  --alignment_ckpt "${BOOTSTRAP_ALIGNMENT_CKPT}" \
  --num_episodes "${COLLECT_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --output_root "${COLLECT_ROOT}" \
  --name_suffix "${COLLECT_SUFFIX}" \
  --target_provider_mode teacher_oracle \
  --planner_no_depth --planner_no_force --no_depth --no_force \
  --record_video --write_episode_videos --no_best_gif \
  --run_full_horizon_on_success \
  --support_states_output_npz "${SUPPORT_NPZ}" \
  --eval_seed "${SEED}" 2>&1 | tee -a "${LOG_FILE}"

echo "[pipeline] 2/4 build phase-1 visual candidate dataset" | tee -a "${LOG_FILE}"
/home/guoning/my_conda_envs/vla-adapter/bin/python scripts/build_pose_candidate_dataset.py \
  --input_dir "${SUPPORT_DIR}" \
  --output_path "${DATASET_NPZ}" \
  --ready_label_mode teacher_ready_or_handoff \
  --phase1_truncate_to_first_success \
  --phase1_drop_weak_success_episodes \
  --phase1_success_xy_threshold 0.006 \
  --phase1_success_abs_z_threshold 0.005 \
  --phase1_success_yaw_threshold 0.12 \
  --candidate_mode primitives \
  --primitive_include_descend \
  --primitive_include_combos \
  --no_primitive_include_tilt \
  --basin_radius_tilt -1.0 2>&1 | tee -a "${LOG_FILE}"

echo "[pipeline] 3/4 train pure-visual no-target-context student scorer" | tee -a "${LOG_FILE}"
/home/guoning/my_conda_envs/vla-adapter/bin/python scripts/train_pose_field_scorer.py \
  --dataset_npz "${DATASET_NPZ}" \
  --output_dir "${SCORER_DIR}" \
  --epochs 25 \
  --batch_size 64 \
  --lr 0.0008 \
  --target_temperature 0.35 \
  --lambda_ready 0.35 \
  --no_target_context \
  --use_depth_stratified_sampler \
  --stratified_high_fraction 0.30 \
  --stratified_mid_fraction 0.25 \
  --stratified_low_fraction 0.30 \
  --stratified_ready_fraction 0.15 \
  --seed "${SEED}" 2>&1 | tee -a "${LOG_FILE}"

echo "[pipeline] 4/4 benchmark planner / teacher upper bound / visual student (${BENCH_EPISODES} episodes)" | tee -a "${LOG_FILE}"
xvfb-run -a /home/guoning/my_conda_envs/vla-adapter/bin/python scripts/evaluate_rlbench_modes.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --task_name "${TASK_NAME}" \
  --modes planner_only,oracle_executed_pregrasp_collect,visual_scorer_mainline \
  --alignment_ckpt "${STUDENT_CKPT}" \
  --num_episodes "${BENCH_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --output_root "${BENCH_ROOT}" \
  --name_suffix "seed${SEED}_phase1_yaw_student_benchmark_mp4" \
  --target_provider_mode teacher_oracle \
  --planner_no_depth --planner_no_force --no_depth --no_force \
  --record_video --write_episode_videos --no_best_gif \
  --run_full_horizon_on_success \
  --eval_seed "${SEED}" 2>&1 | tee -a "${LOG_FILE}"

echo "[pipeline] done" | tee -a "${LOG_FILE}"
