#!/usr/bin/env python3
"""Export a frozen merged VLA state for faster/reproducible RLBench eval.

This performs the expensive base-VLM + LoRA merge once, then writes
`merged_eval_vla_state.pt` under the checkpoint directory. Subsequent
`evaluate_rlbench.py` runs can load that state directly and skip adapter merge.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--vlm_path", default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b")
    parser.add_argument("--config_path", default="pretrained_models/configs/config.json")
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--use_force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "scripts"))
    os.environ.setdefault("VLA_PLATFORM", "RLBENCH")
    os.environ["EVAL_USE_MERGED_STATE"] = "0"
    os.environ["EVAL_MERGE_LORA"] = "1"

    from evaluate_rlbench import load_checkpoint  # noqa: WPS433

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output_path) if args.output_path else checkpoint_dir / "merged_eval_vla_state.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vla, _processor, _action_head, _proprio_projector, _norm_stats = load_checkpoint(
        checkpoint_dir,
        vlm_path=args.vlm_path,
        config_path=args.config_path,
        use_depth=bool(args.use_depth),
        use_force=bool(args.use_force),
    )
    vla_core = getattr(getattr(vla, "base_model", None), "model", vla)
    state = {k: v.detach().cpu() for k, v in vla_core.state_dict().items()}
    torch.save(
        {
            "format": "merged_eval_vla_state_v1",
            "checkpoint_dir": str(checkpoint_dir),
            "model_state_dict": state,
        },
        output_path,
    )
    print(f"[export_merged_eval_vla_state] saved {output_path}")


if __name__ == "__main__":
    main()
