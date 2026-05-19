#!/bin/bash
# Smoke test: insert_onto_square_peg with depth + force, 50 steps on single GPU
export WANDB_MODE=offline
export WANDB_CONSOLE=off
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate vla-adapter

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --config_file_path pretrained_models/configs \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --use_minivlm True \
  --use_rlbench True \
  --rlbench_data_root data/rlbench_data \
  --rlbench_task_name insert_onto_square_peg \
  --use_l1_regression True \
  --use_proprio True \
  --use_depth True \
  --use_force True \
  --use_pro_version True \
  --use_lora True \
  --lora_rank 64 \
  --num_images_in_input 2 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --max_steps 50 \
  --save_freq 25 \
  --num_steps_before_decay 40 \
  --lr_warmup_steps 5 \
  --image_aug True \
  --run_root_dir outputs/smoke_test_insert \
  --run_id_note smoke_depth_force \
  --wandb_project vla_smoke_test \
  --wandb_log_freq 5 \
  --phase Training
