"""
evaluate_rlbench_modes.py

Run several RLBench eval modes while reusing a single loaded VLA planner.
This is intended for small ablations where repeated DINO/SigLIP/Qwen loading
dominates runtime.
"""

import argparse
import copy
import os
from pathlib import Path

from evaluate_rlbench import evaluate, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple NFCR modes with one planner load")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="insert_onto_square_peg")
    parser.add_argument("--modes", type=str, default="planner_only")
    parser.add_argument("--num_episodes", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output_root", type=str, default="eval_logs/insert_onto_square_peg")
    parser.add_argument("--name_suffix", type=str, default="s3407")
    parser.add_argument("--alignment_ckpt", type=str, default=None)
    parser.add_argument("--contact_ckpt", type=str, default=None)
    parser.add_argument("--vlm_path", type=str, default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")
    parser.add_argument("--config_path", type=str, default="pretrained_models/configs/config.json")
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--no_depth", dest="use_depth", action="store_false")
    parser.add_argument("--use_force", action="store_true", default=True)
    parser.add_argument("--no_force", dest="use_force", action="store_false")
    parser.add_argument("--planner_use_depth", dest="planner_use_depth", action="store_true")
    parser.add_argument("--planner_no_depth", dest="planner_use_depth", action="store_false")
    parser.add_argument("--planner_use_force", dest="planner_use_force", action="store_true")
    parser.add_argument("--planner_no_force", dest="planner_use_force", action="store_false")
    parser.add_argument("--record_video", action="store_true", default=True)
    parser.add_argument("--no_video", dest="record_video", action="store_false")
    parser.add_argument("--depth_max", type=float, default=1.0)
    parser.add_argument("--invalid_recovery_lift", type=float, default=0.008)
    parser.add_argument("--invalid_recovery_lift_after", type=int, default=1)
    parser.add_argument("--workspace_clamp_mode", type=str, default="diagnostic", choices=["off", "diagnostic", "tolerance", "hard"])
    parser.add_argument("--workspace_clamp_tolerance", type=float, default=0.01)
    parser.add_argument("--alignment_gate_lookahead", type=int, default=4)
    parser.add_argument("--require_close_intent_for_alignment", dest="require_close_intent_for_alignment", action="store_true", default=True)
    parser.add_argument("--allow_alignment_without_close_intent", dest="require_close_intent_for_alignment", action="store_false")
    parser.add_argument("--enable_alignment_pose", action="store_true", default=True)
    parser.add_argument("--disable_alignment_pose", dest="enable_alignment_pose", action="store_false")
    parser.add_argument("--use_pose_alpha", action="store_true", default=True)
    parser.add_argument("--disable_pose_alpha", dest="use_pose_alpha", action="store_false")
    parser.add_argument("--enable_readiness_gripper", action="store_true", default=False)
    parser.add_argument("--readiness_close_threshold", type=float, default=0.65)
    parser.add_argument("--gripper_override_confidence", type=float, default=0.60)
    parser.add_argument("--record_gripper_trace", action="store_true", default=True)
    parser.add_argument("--no_gripper_trace", dest="record_gripper_trace", action="store_false")
    parser.add_argument("--gripper_close_threshold", type=float, default=0.5)
    parser.add_argument("--gripper_open_threshold", type=float, default=0.8)
    parser.add_argument("--gripper_near_depth_threshold", type=float, default=0.08)
    parser.add_argument("--gripper_close_lookahead", type=int, default=4)
    parser.add_argument("--gripper_min_close_votes", type=int, default=2)
    parser.add_argument("--gripper_min_hold_steps", type=int, default=40)
    parser.add_argument("--gripper_release_open_votes", type=int, default=3)
    parser.add_argument("--gripper_allow_release", action="store_true", default=True)
    parser.add_argument("--gripper_no_release", dest="gripper_allow_release", action="store_false")
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.set_defaults(planner_use_depth=False, planner_use_force=False)
    return parser.parse_args()


def build_mode_args(base_args, mode):
    args = copy.copy(base_args)
    args.stage_refiner_mode = mode
    args.use_stage_aware_refiner = mode not in ("planner_only", "gripper")
    args.use_gripper_supervisor = mode in ("gripper", "alignment_gripper")
    args.use_rule_reflex = False
    args.use_learned_residual = False
    args.residual_ckpt = None

    if mode == "planner_only":
        args.alignment_ckpt = None
        args.contact_ckpt = None
    elif mode == "gripper":
        args.stage_refiner_mode = "planner_only"
        args.alignment_ckpt = None
        args.contact_ckpt = None
    elif mode == "safety_only":
        args.alignment_ckpt = None
        args.contact_ckpt = None
        args.enable_alignment_pose = False
        args.enable_readiness_gripper = False
    elif mode in ("alignment", "pose_alignment_only", "readiness_alignment_pose_only"):
        if not args.alignment_ckpt:
            raise ValueError("alignment mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_readiness_gripper = False
    elif mode in ("readiness_only", "basin_only"):
        if not args.alignment_ckpt:
            raise ValueError(f"{mode} mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = False
        args.enable_readiness_gripper = True
        args.use_pose_alpha = False
    elif mode in ("pose_alignment_only_v2", "pose_alignment_only_basin"):
        if not args.alignment_ckpt:
            raise ValueError(f"{mode} mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = True
        args.enable_readiness_gripper = False
        args.use_pose_alpha = False
    elif mode in ("readiness_pose_v2", "basin_pose_joint"):
        if not args.alignment_ckpt:
            raise ValueError(f"{mode} mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.stage_refiner_mode = "alignment"
        args.enable_alignment_pose = True
        args.enable_readiness_gripper = True
        args.use_pose_alpha = False
    elif mode == "alignment_gripper":
        args.stage_refiner_mode = "alignment"
        if not args.alignment_ckpt:
            raise ValueError("alignment_gripper mode requires --alignment_ckpt")
        args.contact_ckpt = None
    elif mode == "readiness_alignment_full":
        args.stage_refiner_mode = "alignment"
        if not args.alignment_ckpt:
            raise ValueError("readiness_alignment_full mode requires --alignment_ckpt")
        args.contact_ckpt = None
        args.use_gripper_supervisor = False
        args.enable_readiness_gripper = True
    elif mode == "contact":
        if not args.contact_ckpt:
            raise ValueError("contact mode requires --contact_ckpt")
        args.alignment_ckpt = None
    elif mode == "full":
        if not args.alignment_ckpt or not args.contact_ckpt:
            raise ValueError("full mode requires --alignment_ckpt and --contact_ckpt")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    base_name = f"insert_vo40k_{mode}_{args.name_suffix}"
    args.output_dir = str(Path(args.output_root) / base_name)
    return args


def main():
    args = parse_args()
    os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
    if args.planner_use_depth is None:
        args.planner_use_depth = args.use_depth
    if args.planner_use_force is None:
        args.planner_use_force = args.use_force

    print("[multi_eval] Loading planner once...")
    preloaded = load_checkpoint(
        args.checkpoint_dir,
        vlm_path=args.vlm_path,
        config_path=args.config_path,
        use_depth=args.planner_use_depth,
        use_force=args.planner_use_force,
    )

    summaries = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        mode_args = build_mode_args(args, mode)
        print(f"\n[multi_eval] Running mode={mode} -> {mode_args.output_dir}")
        summaries[mode] = evaluate(mode_args, preloaded_components=preloaded)

    print("\n[multi_eval] Done.")
    for mode, success_rate in summaries.items():
        print(f"  {mode}: success_rate={success_rate:.3f}")


if __name__ == "__main__":
    main()
