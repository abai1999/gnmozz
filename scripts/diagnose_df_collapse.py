"""
Diagnose depth+force model collapse across checkpoints.
Checks: gate values, adapter output magnitudes, branch contributions, per-dim predictions.
"""
import sys, os, glob, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM, NUM_TOKENS


def load_action_head(ckpt_dir):
    """Load action head state dict from checkpoint."""
    # Find action_head file (may have step suffix)
    candidates = glob.glob(os.path.join(ckpt_dir, "action_head*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No action_head*.pt in {ckpt_dir}")
    ah_path = candidates[0]
    return torch.load(ah_path, map_location="cpu")


def analyze_gates(state_dict, label):
    """Print all gating values."""
    print(f"\n{'='*60}")
    print(f"  Checkpoint: {label}")
    print(f"{'='*60}")
    
    gate_keys = sorted([k for k in state_dict if 'gating' in k])
    
    # Group by block
    blocks = {}
    for k in gate_keys:
        # e.g. model.mlp_resnet_blocks.0.gating_factor
        parts = k.split('.')
        block_idx = None
        gate_name = None
        for i, p in enumerate(parts):
            if p == 'mlp_resnet_blocks':
                block_idx = int(parts[i+1])
            if 'gating' in p:
                gate_name = p
        if block_idx is not None and gate_name is not None:
            if block_idx not in blocks:
                blocks[block_idx] = {}
            val = state_dict[k].item()
            blocks[block_idx][gate_name] = val
    
    print(f"\n--- Gating values (raw → tanh) per block ---")
    print(f"{'Block':>5} | {'gating_factor':>20} | {'gating_depth':>20} | {'gating_force':>20}")
    print("-" * 75)
    for idx in sorted(blocks.keys()):
        row = []
        for name in ['gating_factor', 'gating_depth', 'gating_force']:
            if name in blocks[idx]:
                raw = blocks[idx][name]
                tanh_val = np.tanh(raw)
                row.append(f"{raw:+.4f} → {tanh_val:+.4f}")
            else:
                row.append("N/A")
        print(f"{idx:>5} | {row[0]:>20} | {row[1]:>20} | {row[2]:>20}")
    
    # Summary stats
    for gate_name in ['gating_factor', 'gating_depth', 'gating_force']:
        vals = [blocks[b][gate_name] for b in blocks if gate_name in blocks[b]]
        if vals:
            tanh_vals = [np.tanh(v) for v in vals]
            print(f"\n  {gate_name}: raw mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                  f"tanh mean={np.mean(tanh_vals):.4f}, max={np.max(tanh_vals):.4f}, min={np.min(tanh_vals):.4f}")


def analyze_adapter_weights(state_dict, label):
    """Check depth/force adapter projection weights."""
    print(f"\n--- Adapter projection layer norms ---")
    for key_substr in ['depth_adapter.proj', 'force_adapter.proj']:
        for suffix in ['.weight', '.bias']:
            full_pattern = key_substr + suffix
            matching = [k for k in state_dict if full_pattern in k]
            for k in matching:
                w = state_dict[k]
                print(f"  {k}: norm={w.norm():.6f}, mean={w.mean():.6f}, std={w.std():.6f}, max={w.abs().max():.6f}")


def analyze_o_proj_norms(state_dict, label):
    """Check output projection norms for each branch."""
    print(f"\n--- o_proj weight norms per branch (first 3 + last 3 blocks) ---")
    for proj_name in ['o_proj.weight', 'o_proj_task.weight', 'o_proj_depth.weight', 'o_proj_force.weight']:
        matching = sorted([k for k in state_dict if k.endswith(proj_name)])
        if not matching:
            continue
        norms = [state_dict[k].norm().item() for k in matching]
        print(f"  {proj_name}: blocks[0:3]={[f'{n:.2f}' for n in norms[:3]]}, "
              f"blocks[-3:]={[f'{n:.2f}' for n in norms[-3:]]}, "
              f"mean={np.mean(norms):.2f}, std={np.std(norms):.2f}")


def analyze_fc2_bias(state_dict, label):
    """Check final layer bias — if collapsed, all dims will have similar bias."""
    fc2_keys = [k for k in state_dict if 'fc2' in k and 'bias' in k]
    for k in fc2_keys:
        b = state_dict[k]
        dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
        print(f"\n--- Final fc2 bias (action mean prediction) ---")
        for i, name in enumerate(dim_names):
            print(f"  {name}: {b[i].item():.6f}")


def run_inference_diagnostic(ckpt_dir, gpu_id=0):
    """Load full model and run inference on a few data samples to check branch contributions."""
    from prismatic.models.action_heads import L1RegressionActionHead
    
    # Load action head
    ah_state = load_action_head(ckpt_dir)
    
    # Determine config
    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    
    # Create action head
    head = L1RegressionActionHead(
        input_dim=896,
        hidden_dim=896,
        action_dim=7,
        num_task_tokens=729,  # dinosiglip-224px: ~729 patches
        use_pro_version=True,
        use_depth=True,
        use_force=True,
    )
    head.load_state_dict(ah_state, strict=False)
    head = head.to(f"cuda:{gpu_id}").to(torch.bfloat16).eval()
    
    # Create synthetic input
    B = 1
    num_layers = 25  # Qwen2.5-0.5B has 24 layers + 1 embedding
    num_patches = 729
    hidden_dim = 896
    
    torch.manual_seed(42)
    
    # Simulate hidden states: [task_patches, action_tokens]
    task_states = torch.randn(B, num_layers, num_patches, hidden_dim, dtype=torch.bfloat16, device=f"cuda:{gpu_id}") * 0.1
    action_states = torch.randn(B, num_layers, NUM_TOKENS, hidden_dim, dtype=torch.bfloat16, device=f"cuda:{gpu_id}") * 0.1
    combined = torch.cat([task_states, action_states], dim=2)
    
    proprio = torch.randn(B, 7, dtype=torch.float32, device=f"cuda:{gpu_id}") * 0.1  # PROPRIO_DIM=7
    proprio_proj = torch.nn.Linear(7, hidden_dim).to(f"cuda:{gpu_id}").to(torch.bfloat16)
    
    depth = torch.randn(B, 1, 224, 224, dtype=torch.bfloat16, device=f"cuda:{gpu_id}") * 0.5 + 0.5
    force = torch.randn(B, 6, 32, dtype=torch.bfloat16, device=f"cuda:{gpu_id}") * 0.1
    
    with torch.no_grad():
        # Run WITH depth+force
        out_df = head.predict_action(combined, proprio=proprio, proprio_projector=proprio_proj,
                                      phase="Inference", depth=depth, force_history=force)
        
        # Run WITHOUT depth+force (set to None)
        out_vo = head.predict_action(combined, proprio=proprio, proprio_projector=proprio_proj,
                                      phase="Inference", depth=None, force_history=None)
    
    print(f"\n--- Predictions with vs without depth+force ---")
    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
    print(f"{'dim':>8} | {'with_df':>12} | {'without_df':>12} | {'delta':>12}")
    print("-" * 55)
    for step in [0]:  # First action step
        for i, name in enumerate(dim_names):
            v_df = out_df[0, step, i].item()
            v_vo = out_vo[0, step, i].item()
            delta = v_df - v_vo
            print(f"{name:>8} | {v_df:>12.6f} | {v_vo:>12.6f} | {delta:>+12.6f}")
    
    print(f"\n  With DF:    all-dim mean abs = {out_df.abs().mean():.6f}")
    print(f"  Without DF: all-dim mean abs = {out_vo.abs().mean():.6f}")
    print(f"  XY mag with DF:    {out_df[:,:,:2].abs().mean():.6f}")
    print(f"  XY mag without DF: {out_vo[:,:,:2].abs().mean():.6f}")


def check_data_statistics(ckpt_dir):
    """Check dataset statistics for depth/force normalization."""
    stats_path = os.path.join(ckpt_dir, "dataset_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"\n--- Dataset statistics ---")
        for k, v in stats.items():
            if isinstance(v, list):
                print(f"  {k}: {[f'{x:.4f}' for x in v]}")
            else:
                print(f"  {k}: {v}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="outputs/insert_long_train")
    parser.add_argument("--run_name", default="insert_df_v3_50k")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", nargs="+", type=int, default=[5000, 10000, 15000])
    args = parser.parse_args()
    
    base_pattern = None
    # Find matching run dirs
    for d in sorted(os.listdir(args.run_dir)):
        if args.run_name in d and not d.endswith("_chkpt"):
            base_pattern = d
            break
    
    if base_pattern is None:
        print(f"ERROR: Cannot find run dir matching '{args.run_name}' in {args.run_dir}")
        return
    
    print(f"Base run: {base_pattern}")
    
    for step in args.steps:
        ckpt_dir = os.path.join(args.run_dir, f"{base_pattern}--{step}_chkpt")
        if not os.path.exists(ckpt_dir):
            print(f"\nSkipping step {step}: {ckpt_dir} not found")
            continue
        
        state_dict = load_action_head(ckpt_dir)
        label = f"step {step}"
        
        analyze_gates(state_dict, label)
        analyze_adapter_weights(state_dict, label)
        analyze_o_proj_norms(state_dict, label)
        analyze_fc2_bias(state_dict, label)
        check_data_statistics(ckpt_dir)
    
    # Run inference diagnostic on last checkpoint
    last_step = max(args.steps)
    last_ckpt = os.path.join(args.run_dir, f"{base_pattern}--{last_step}_chkpt")
    if os.path.exists(last_ckpt):
        print(f"\n{'='*60}")
        print(f"  Inference diagnostic @ step {last_step}")
        print(f"{'='*60}")
        run_inference_diagnostic(last_ckpt, args.gpu)


if __name__ == "__main__":
    main()
