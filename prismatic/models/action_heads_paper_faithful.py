"""
action_heads_paper_faithful.py

Paper-faithful planner core for controlled regression experiments.

This mirrors the original VLA-Adapter-main L1 action head logic:
- zero-initialized action latent
- original MLPResNetBlock_Pro bridge structure

The public constructor mirrors the current project action head so we can swap
cores without changing the RLBench training/eval scaffolding.
"""

import math

import torch
import torch.nn as nn

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations


class PaperFaithfulL1RegressionActionHead(nn.Module):
    """Original VLA-Adapter L1 policy core with a compatibility-friendly API."""

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
        use_depth=False,
        use_force=False,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_depth = use_depth
        self.use_force = use_force
        self.model = MLPResNet(
            num_blocks=24,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            use_pro_version=use_pro_version,
        )

    def predict_action(
        self,
        actions_hidden_states,
        proprio=None,
        proprio_projector=None,
        phase="Inference",
        depth=None,
        force_history=None,
        **kwargs,
    ):
        del depth, force_history, kwargs
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)
        proprio_features = proprio_projector(proprio)
        proprio_features = proprio_features.unsqueeze(dim=1)

        task_hidden_states = actions_hidden_states[:, :, : self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens :, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device,
            dtype=actions_hidden_states.dtype,
        ).detach()
        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )

        if phase == "Training":
            _, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(
                seq_len,
                dim,
                device=rearranged_actions_hidden_states.device,
                dtype=rearranged_actions_hidden_states.dtype,
            )
            rearranged_actions_hidden_states = rearranged_actions_hidden_states + random_perturbations

        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
        )
        return action


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim, use_pro_version=False):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_a=None, h_t=None, p=None):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class MLPResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.k_h_t = nn.Linear(dim, dim)
        self.v_h_t = nn.Linear(dim, dim)
        self.k_h_a_p = nn.Linear(dim, dim)
        self.v_h_a_p = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim * 3, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x, h_t=None, h_a=None, p=None):
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        h_a_p = torch.cat((h_a, p), dim=1)
        q = self.q_proj(x)
        k_x = self.k_proj(x)
        v_x = self.v_proj(x)
        k_h_t = self.k_h_t(h_t)
        v_h_t = self.v_h_t(h_t)
        k_h_a_p = self.k_h_a_p(h_a_p)
        v_h_a_p = self.v_h_a_p(h_a_p)

        attn_scores_x = torch.matmul(q, k_x.transpose(-2, -1))
        attn_weights_x = torch.softmax(attn_scores_x, dim=-1)
        output_x = torch.matmul(attn_weights_x, v_x)

        attn_scores_t = torch.matmul(q, k_h_t.transpose(-2, -1))
        attn_weights_t = torch.softmax(attn_scores_t, dim=-1)
        output_t = torch.matmul(attn_weights_t, v_h_t) * ratio_g

        attn_scores_a = torch.matmul(q, k_h_a_p.transpose(-2, -1))
        attn_weights_a = torch.softmax(attn_scores_a, dim=-1)
        output_a = torch.matmul(attn_weights_a, v_h_a_p)

        output = torch.cat((output_x, output_t, output_a), dim=-1)
        output = self.o_proj(output)
        x = self.ffn(output + x)
        return x


class MLPResNetBlock_Pro(nn.Module):
    """Original VLA-Adapter Pro bridge block."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))
        self.rope = RotaryPositionEmbedding(self.head_dim)
        self.film_gen = nn.Sequential(nn.Linear(dim, dim * 2))

    def apply_film(self, x, gamma, beta):
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)

    def forward(self, x, h_a=None, h_t=None, p=None):
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        h_adapter = torch.cat((h_a, p), dim=1)
        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        q_1 = self.q_proj(x)
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)

        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        v_list = [v_tokens, v_adapter, v_task]

        concat_scores = torch.cat(attn_scores, dim=-1)
        concat_weights = torch.softmax(concat_scores / math.sqrt(self.head_dim), dim=-1)
        split_sizes = [s.size(-1) for s in attn_scores]
        split_weights = torch.split(concat_weights, split_sizes, dim=-1)
        outputs = [torch.matmul(w, v) for w, v in zip(split_weights, v_list)]
        output = sum(outputs)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        x = self.ffn(output + x)
        return x
