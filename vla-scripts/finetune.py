"""
finetune.py

Fine-tunes Qwen2.5-0.5B via LoRA.
"""

import os
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import numpy as np
import torch.nn.functional as F
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

os.environ.setdefault("VLA_PLATFORM", "RLBENCH")

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.action_heads_paper_faithful import PaperFaithfulL1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, save_dataset_statistics
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS
)
from prismatic.vla.datasets.dagger_rlbench_dataset import DaggerRLBenchDataset
from prismatic.vla.datasets.mixed_rlbench_dataset import MixedRLBenchDataset
from prismatic.vla.datasets.rlbench_dataset import RLBenchDataset
from prismatic.models import load, load_vla



# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    use_minivlm: bool = False                        #
    resum_vla_path: str = "openvla/openvla-7b"       # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # RLBench Dataset
    rlbench_data_root: str = "data/rlbench_data"             # Root directory containing RLBench episode data
    rlbench_task_name: str = "insert_onto_square_peg"        # RLBench task name (subfolder name)
    rlbench_sample_weights_path: Optional[str] = None        # Optional per-frame sample weights (.npy or .npz)
    dagger_data_dir: Optional[str] = None                    # Optional planner-state DAgger shards directory
    dagger_mix_expert: int = 2                               # Expert repeat factor when mixing DAgger data
    dagger_mix_dagger: int = 1                               # DAgger repeat factor when mixing DAgger data
    dagger_oversample_align: int = 3                         # DAgger ALIGN-stage oversampling
    dagger_oversample_interact: int = 2                      # DAgger INTERACT/RECOVER oversampling
    dagger_oversample_transition: int = 4                    # DAgger phase-transition oversampling
    dagger_oversample_failure: int = 5                       # DAgger failure / no-progress oversampling
    run_root_dir: Path = Path("runs")                        # Path to directory to store logs & checkpoints

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images (front + wrist)
    use_proprio: bool = True                         # If True, includes robot proprioceptive state in input
    use_depth: bool = False                          # If True, include wrist depth in action head
    use_force: bool = False                          # If True, include force history in action head
    force_history_len: int = 32                      # Number of past force readings to include
    planner_core_variant: str = "current"            # current | paper_faithful

    # Training configuration
    batch_size: int = 4                              # Batch size per device
    learning_rate: float = 2e-4                      # Learning rate
    lr_warmup_steps: int = 0.1                       # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100000             # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 50000                           # Max number of training steps
    save_freq: int = 5000                            # Checkpoint saving frequency in steps
    save_steps: Optional[str] = None                  # Optional comma-separated exact checkpoint steps
    save_latest_checkpoint_only: bool = False         # If True, saves only 1 checkpoint
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations
    seed: Optional[int] = None                      # If set, make training stochasticity reproducible
    dataloader_num_workers: int = 4                  # Number of dataloader workers
    dataloader_pin_memory: bool = False              # Pin CPU memory for faster host->device transfer
    dataloader_persistent_workers: bool = False      # Keep workers alive across iterations
    grad_clip_norm: float = 1.0                      # Clip grad norm after backward; <=0 disables
    abort_on_nonfinite_loss: bool = True             # Abort training immediately on NaN/Inf loss

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 64                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights during training

    # Full Finetune
    use_fz: bool = False                             # If True, freezes backbone (non-LoRA mode)

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # revision version
    use_pro_version: bool = True                             # the version number
    phase: str = "Training"
    # fmt: on



def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict



def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.config_file_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.config_file_path.split('/')[-1]}+{cfg.rlbench_task_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def set_training_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set Python / NumPy / PyTorch RNG seeds for reproducible training.

    Args:
        seed (int): Base random seed.
        deterministic (bool): If True, prefer deterministic CUDA kernels where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def load_sample_weights(path: str) -> np.ndarray:
    """Load per-frame sample weights from .npy or .npz."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"sample weights not found: {p}")
    if p.suffix == ".npy":
        weights = np.load(p)
    elif p.suffix == ".npz":
        z = np.load(p)
        key = "sample_weights" if "sample_weights" in z.files else z.files[0]
        weights = z[key]
    else:
        raise ValueError(f"unsupported sample-weights format: {p.suffix}")
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(weights)):
        raise ValueError("sample weights contain non-finite values")
    if np.any(weights < 0.0):
        raise ValueError("sample weights must be non-negative")
    return weights


def trainable_parameters_for_clipping(
    vla: nn.Module,
    action_head: nn.Module,
    proprio_projector: Optional[nn.Module],
) -> list[torch.nn.Parameter]:
    params = list(p for p in vla.parameters() if p.requires_grad)
    params.extend(p for p in action_head.parameters() if p.requires_grad)
    if proprio_projector is not None:
        params.extend(p for p in proprio_projector.parameters() if p.requires_grad)
    return params



def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)



def parse_save_steps(save_steps: Optional[str], max_steps: int) -> Optional[set[int]]:
    if save_steps is None:
        return None
    parsed_steps = set()
    for chunk in str(save_steps).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        step = int(chunk)
        if step <= 0:
            raise ValueError(f"save_steps must contain positive integers, got {step}")
        if step > max_steps:
            raise ValueError(f"save_steps contains step {step} beyond max_steps={max_steps}")
        parsed_steps.add(step)
    return parsed_steps or None


def should_save_checkpoint(cfg: FinetuneConfig, log_step: int, exact_save_steps: Optional[set[int]]) -> bool:
    if log_step <= 0:
        return False
    if exact_save_steps is not None:
        return log_step in exact_save_steps
    return cfg.save_freq > 0 and log_step % cfg.save_freq == 0


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

    print(f"# trainable params in {name}: {num_params}")



def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict, strict=False)
        print('loaded!!!!!!!!!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def build_action_head_module(cfg: FinetuneConfig, llm_dim: int) -> tuple[Type[nn.Module], dict]:
    if cfg.planner_core_variant == "paper_faithful":
        action_head_class = PaperFaithfulL1RegressionActionHead
    else:
        action_head_class = L1RegressionActionHead
    return (
        action_head_class,
        {
            "input_dim": llm_dim,
            "hidden_dim": llm_dim,
            "action_dim": ACTION_DIM,
            "use_pro_version": cfg.use_pro_version,
            "use_depth": cfg.use_depth,
            "use_force": cfg.use_force,
        },
    )



def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None,
    use_depth=False,
    use_force=False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"].to(device_id),
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
            )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:,1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches+1:].argmax(dim=2)

        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids,
            ground_truth_token_ids,
            mask=current_action_mask
            )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer,
            predicted_token_ids,
            ground_truth_token_ids,
            mask=current_action_mask
            )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids,
            ground_truth_token_ids,
            mask=next_actions_mask
            )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer,
            predicted_token_ids,
            ground_truth_token_ids,
            mask=next_actions_mask
            )

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Compute metrics for continuous action representations (L1 regression)
    else:
        # Get last layer hidden states
        multi_layer_hidden_states = []

        for item in output.hidden_states[0:]:
            # Get hidden states after BOS + vision patches (aligned with labels[:, 1:])
            text_hidden_states = item[:, num_patches+1:]
            # Get hidden states for action portion of response
            batch_size = batch["input_ids"].shape[0]
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, 1,NUM_TOKENS, -1).to(torch.bfloat16)
            task_latten_states = item[:, 1:num_patches+1].reshape(batch_size, 1, num_patches , -1)
            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states),2)
            multi_layer_hidden_states.append(all_hidden_states)
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)

        predicted_actions = action_head.module.predict_action(
            multi_layer_hidden_states,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            phase=cfg.phase,
            depth=batch["wrist_depth"].to(torch.bfloat16).to(device_id) if use_depth and "wrist_depth" in batch else None,
            force_history=batch["force_history"].to(torch.bfloat16).to(device_id) if use_force and "force_history" in batch else None,
            )

        if not torch.isfinite(predicted_actions).all():
            raise FloatingPointError("predicted_actions became non-finite")

        loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)

        if not torch.isfinite(loss):
            raise FloatingPointError("continuous-action loss became non-finite")

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )

        # Get detailed L1 losses for logging
        should_log_l1_loss = use_l1_regression
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            if compute_diffusion_l1:
                print('curr: ',curr_action_l1_loss.item())

            # Per-dimension L1 losses (all timesteps)
            dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
            per_dim_l1 = torch.nn.functional.l1_loss(
                predicted_actions, ground_truth_actions, reduction='none'
            ).mean(dim=(0, 1))  # (7,)
            per_dim_metrics = {f"L1_{name}": per_dim_l1[i].item() for i, name in enumerate(dim_names)}
            # xy magnitude: mean predicted |dx,dy| across batch and timesteps
            xy_mag = predicted_actions[:, :, :2].abs().mean().item()
            per_dim_metrics["xy_mag"] = xy_mag

            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                    **per_dim_metrics,
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics



def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics



def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)



def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,

) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        planner_head_config = {
            "planner_core_variant": str(getattr(cfg, "planner_core_variant", "current")),
            "action_head_type": (
                "paper_faithful_l1_resnet"
                if str(getattr(cfg, "planner_core_variant", "current")) == "paper_faithful"
                else "l1_resnet"
            ),
            "use_depth": bool(getattr(cfg, "use_depth", False)),
            "use_force": bool(getattr(cfg, "use_force", False)),
            "use_pro_version": bool(getattr(cfg, "use_pro_version", False)),
        }
        with open(checkpoint_dir / "planner_head_config.json", "w") as f:
            json.dump(planner_head_config, f, indent=2)

        if cfg.use_fz:
            vla.module.save_pretrained(checkpoint_dir) # directly save checkpoint without lora
        else:
            vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_l1_regression and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        # Save learned action_queries (they are unfrozen and trained, but NOT part of LoRA adapter)
        # Search for any key ending with action_queries.weight (robust to wrapper changes)
        aq_state = {}
        for key, val in vla.state_dict().items():
            if key.endswith("action_queries.weight"):
                aq_state["action_queries.weight"] = val.cpu()
                break
        if aq_state:
            torch.save(aq_state, checkpoint_dir / f"action_queries--{checkpoint_name_suffix}")
        else:
            print("[WARNING] action_queries.weight not found in state_dict ? not saved!")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        if cfg.use_minivlm:
            config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
            base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)  # Create a new model with configuration, the parameters are randomly initialized
            # print(new_state_dict['action_queries.weight'])
            new_state_dict['action_queries.weight'] = vla.state_dict()['module.base_model.model.action_queries.weight'].cpu()
            missing_keys, unexpected_keys = base_vla.load_state_dict(new_state_dict, strict=False)

        else:
            base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
        )


        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # Wait for merged model to be saved
        dist.barrier()



def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
                use_depth=cfg.use_depth,
                use_force=cfg.use_force,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """

    global RAW_STATE_DICT

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.config_file_path}` on `{cfg.rlbench_task_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    exact_save_steps = parse_save_steps(cfg.save_steps, cfg.max_steps)
    if exact_save_steps is not None:
        print(f"Using explicit checkpoint save steps: {sorted(exact_save_steps)}")

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if cfg.seed is not None:
        process_seed = int(cfg.seed) + int(distributed_state.process_index)
        set_training_seed(process_seed)
        print(f"Using fixed training seed: base={cfg.seed}, process_seed={process_seed}")

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="offline")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect

    if model_is_on_hf_hub(cfg.config_file_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        # Overwrite VLA path
        cfg.config_file_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path)
        check_model_logic_mismatch(cfg.config_file_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)

    if cfg.use_minivlm:
        hf_token = ''
        if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:

            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(
                cfg.vlm_path,
                hf_token=hf_token,
                load_for_training=True,
                )
        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)  # Create a new model with configuration, the parameters are randomly initialized
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
            ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict

        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)

        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict

    else:
        RAW_STATE_DICT ={}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
            ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # vla.set_version(cfg.version)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha= 2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        if cfg.resume:
            adapter_dir = Path(cfg.resum_vla_path) / "lora_adapter"
            if not adapter_dir.exists():
                raise FileNotFoundError(f"LoRA adapter not found at {adapter_dir}")
            vla = PeftModel.from_pretrained(vla, adapter_dir, is_trainable=True)
            print(f"Loaded LoRA adapter from {adapter_dir}")
        else:
            vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()

    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.config_file_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head_class, action_head_kwargs = build_action_head_module(cfg, vla.module.llm_dim)
        action_head = init_module(
        action_head_class,
        "action_head",
        cfg,
        device_id,
        action_head_kwargs,
        to_bf16=True,
        find_unused_params=True,
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]

    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    # 1. MultiStepLR
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )
    # 2. CosineAnnealingLR
    # scheduler = CosineAnnealingLR(
    #         optimizer,
    #         T_max=cfg.num_steps_before_decay,
    #         eta_min=0.0001,
    #         )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Create RLBench map-style dataset
    expert_dataset = RLBenchDataset(
        data_root=cfg.rlbench_data_root,
        task_name=cfg.rlbench_task_name,
        image_transform=processor.image_processor.apply_transform,
        action_tokenizer=action_tokenizer,
        tokenizer=processor.tokenizer,
        prompt_builder_fn=PurePromptBuilder,
        use_depth=cfg.use_depth,
        use_force=cfg.use_force,
        force_history_len=cfg.force_history_len,
        image_aug=cfg.image_aug,
    )

    if cfg.dagger_data_dir:
        dagger_dataset = DaggerRLBenchDataset(
            data_dir=cfg.dagger_data_dir,
            image_transform=processor.image_processor.apply_transform,
            action_tokenizer=action_tokenizer,
            tokenizer=processor.tokenizer,
            prompt_builder_fn=PurePromptBuilder,
            use_depth=cfg.use_depth,
            use_force=cfg.use_force,
            image_aug=cfg.image_aug,
            oversample_align=cfg.dagger_oversample_align,
            oversample_interact=cfg.dagger_oversample_interact,
            oversample_transition=cfg.dagger_oversample_transition,
            oversample_failure=cfg.dagger_oversample_failure,
        )
        train_dataset = MixedRLBenchDataset(
            expert_dataset,
            dagger_dataset,
            expert_repeat=cfg.dagger_mix_expert,
            dagger_repeat=cfg.dagger_mix_dagger,
        )
        print(
            f"[finetune] Using mixed expert+dagger dataset: "
            f"expert={len(expert_dataset)}, dagger={len(dagger_dataset)}, "
            f"ratio={cfg.dagger_mix_expert}:{cfg.dagger_mix_dagger}"
        )
        if hasattr(dagger_dataset, "get_summary"):
            print(f"[finetune] DAgger summary: {dagger_dataset.get_summary()}")
    else:
        train_dataset = expert_dataset

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(expert_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader_generator = None
    worker_init_fn = None
    if cfg.seed is not None:
        data_seed = int(cfg.seed) + int(distributed_state.process_index)
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(data_seed)

        def worker_init_fn(worker_id):
            worker_seed = data_seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    sampler = None
    shuffle = True
    if cfg.rlbench_sample_weights_path is not None:
        sample_weights = load_sample_weights(cfg.rlbench_sample_weights_path)
        if len(sample_weights) != len(train_dataset):
            raise ValueError(
                f"sample_weights length mismatch: got {len(sample_weights)} for dataset length {len(train_dataset)}"
            )
        if float(sample_weights.sum()) <= 0.0:
            raise ValueError("sample_weights sum must be positive")
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=dataloader_generator,
        )
        shuffle = False
        print(
            f"[finetune] Using WeightedRandomSampler from {cfg.rlbench_sample_weights_path}: "
            f"count={len(sample_weights)} min={sample_weights.min():.6f} "
            f"mean={sample_weights.mean():.6f} max={sample_weights.max():.6f}"
        )

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=cfg.dataloader_pin_memory,
        persistent_workers=cfg.dataloader_persistent_workers and cfg.dataloader_num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=dataloader_generator,
    )
    print('Len of dataloader: ', len(dataloader))

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_dx": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_dy": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_dz": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_drx": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_dry": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_drz": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_gripper": deque(maxlen=cfg.grad_accumulation_steps),
        "xy_mag": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    # Wrap dataloader in infinite iterator across epochs (map-style dataset)
    def infinite_dataloader(dl):
        while True:
            for batch in dl:
                yield batch

    data_iter = infinite_dataloader(dataloader)

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        clip_params = trainable_parameters_for_clipping(
            vla=vla,
            action_head=action_head,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
        )
        for batch_idx, batch in enumerate(data_iter):
            # Compute training metrics and loss
            compute_diffusion_l1 = (batch_idx % 50 == 0)  # Log detailed L1 periodically
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
                use_depth=cfg.use_depth,
                use_force=cfg.use_force,
            )

            if cfg.abort_on_nonfinite_loss and not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss detected at batch_idx={batch_idx}, gradient_step_idx={batch_idx // cfg.grad_accumulation_steps}"
                )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=cfg.grad_clip_norm)
                    smoothened_metrics["grad_norm"] = float(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and should_save_checkpoint(cfg, log_step, exact_save_steps):
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                )

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()


