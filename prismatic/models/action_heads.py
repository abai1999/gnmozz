"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS



def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations



class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
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
            input_dim=input_dim*ACTION_DIM, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim,
            use_pro_version=use_pro_version,
            use_depth=use_depth,
            use_force=use_force,
            )

        # Query initializer: project pooled action hidden states → per-step query
        # layer 0 action tokens (64 × hidden_dim) → pool → per-chunk-step query
        self.query_init_proj = nn.Linear(hidden_dim, hidden_dim * ACTION_DIM)
        nn.init.zeros_(self.query_init_proj.weight)
        nn.init.zeros_(self.query_init_proj.bias)

        # Optional depth adapter
        if use_depth:
            from prismatic.models.depth_adapter import DepthAdapter
            self.depth_adapter = DepthAdapter(llm_dim=hidden_dim)

        # Optional force adapter
        if use_force:
            from prismatic.models.force_temporal_adapter import ForceTemporalAdapter
            self.force_adapter = ForceTemporalAdapter(llm_dim=hidden_dim)

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
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        # --- Query initialization from action hidden states (layer 0) ---
        # actions_hidden_states[:, 0, :, :] = (B, NUM_TOKENS=64, hidden_dim)
        action_layer0 = actions_hidden_states[:, 0, :, :]   # (B, 64, hidden_dim)
        # Reshape 64 tokens into NUM_ACTIONS_CHUNK groups, mean-pool each group
        tokens_per_step = action_layer0.shape[1] // NUM_ACTIONS_CHUNK  # 64/8 = 8
        action_pooled = action_layer0.reshape(
            batch_size, NUM_ACTIONS_CHUNK, tokens_per_step, self.hidden_dim
        ).mean(dim=2)  # (B, 8, hidden_dim)
        # Project to query space: (B, 8, hidden_dim) → (B, 8, hidden_dim * action_dim)
        rearranged_actions_hidden_states = self.query_init_proj(action_pooled)  # (B, 8, hidden_dim * action_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

        # Encode optional depth/force modalities
        h_d = None
        if self.use_depth and depth is not None:
            h_d = self.depth_adapter(depth.to(dtype=actions_hidden_states.dtype))  # (B, DEPTH_TOKENS, hidden_dim)

        h_f = None
        if self.use_force and force_history is not None:
            h_f = self.force_adapter(force_history.to(dtype=actions_hidden_states.dtype))  # (B, FORCE_TOKENS, hidden_dim)

        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
            h_d=h_d,
            h_f=h_f,
            )

        return action
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False,
            use_depth=False,
            use_force=False,
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim, use_depth=use_depth, use_force=use_force))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p=None, h_d=None, h_f=None):
 
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t=h_t[:,i+1,:], h_a=h_a[:,i+1,:], p=p, h_d=h_d, h_f=h_f)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x   



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """Multi-modal fusion block with independent branch attention and pre-norm residual.

    Architecture changes vs original:
    1. Self+adapter share one softmax; task/depth/force each have independent attention
       then gated-add back — no cross-branch softmax competition.
    2. Standard pre-norm residual: x = x + attn_out; x = x + ffn(ln(x))
    3. Gate controls *output* contribution, not just logits — gate=0 means truly zero.
    """

    def __init__(self, dim, num_heads=8, use_depth=False, use_force=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_depth = use_depth
        self.use_force = use_force

        # Pre-norm for attention
        self.attn_ln = nn.LayerNorm(dim)

        # FFN with pre-norm (proper residual)
        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn_linear = nn.Linear(dim, dim)
        self.ffn_act = nn.ReLU()

        # Keep old ffn module for checkpoint compat (unused in forward, loaded by state_dict)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V (independent branch)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        # Task branch output projection (default Kaiming init; gate controls magnitude)
        self.o_proj_task = nn.Linear(dim, dim)

        # Depth cross-attention: K, V (optional, independent branch)
        if use_depth:
            self.k_depth = nn.Linear(dim, dim)
            self.v_depth = nn.Linear(dim, dim)
            self.gating_depth = nn.Parameter(torch.full((1,), 0.01))
            self.o_proj_depth = nn.Linear(dim, dim)

        # Force cross-attention: K, V (optional, independent branch)
        if use_force:
            self.k_force = nn.Linear(dim, dim)
            self.v_force = nn.Linear(dim, dim)
            self.gating_force = nn.Parameter(torch.full((1,), 0.01))
            self.o_proj_force = nn.Linear(dim, dim)

        # Main output projection (self + adapter branch)
        self.o_proj = nn.Linear(dim, dim)

        # Post-attention LayerNorm: normalizes the COMBINED attn_out (main + all branches)
        # before adding to residual — prevents scale explosion across branches.
        self.post_attn_ln = nn.LayerNorm(dim)

        # Per-branch LayerNorm on depth/force o_proj output — prevents
        # "small gate × huge raw norm" scale bypass.
        if use_depth:
            self.depth_out_ln = nn.LayerNorm(dim)
        if use_force:
            self.force_out_ln = nn.LayerNorm(dim)

        # Task gating (small positive init so branch gets gradient from step 0)
        self.gating_factor = nn.Parameter(torch.full((1,), 0.01))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM (kept for checkpoint compat, not used) ----
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),
            )


    def _branch_attention(self, q, k, v, scale):
        """Single-branch scaled dot-product attention.
        q: (B, H, T, D), k: (B, H, K, D), v: (B, H, K, D)
        Returns: (B, T, C)
        """
        B = q.size(0)
        T = q.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, K)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # (B, H, T, D)
        return out.transpose(1, 2).contiguous().view(B, T, self.dim)


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation (kept for compat)"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None, h_d=None, h_f=None):
        """
        h_a: adapter tokens (action hidden states from one VLM layer)
        h_t: task tokens (vision hidden states from one VLM layer)
        p:   proprio features
        h_d: depth tokens (optional, from DepthAdapter)
        h_f: force tokens (optional, from ForceTemporalAdapter)
        """
        scale = math.sqrt(self.head_dim)

        # --- Pre-norm ---
        x_norm = self.attn_ln(x)

        # concat adapter and proprio
        h_adapter = torch.cat((h_a, p), dim=1)
        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1)
        K_t = h_task.size(1)

        # Q (shared across branches, computed from normalized x)
        q_1 = self.q_proj(x_norm)

        # --- Main branch: Self + Adapter + Task (shared softmax) ---
        # Vision (task) tokens participate in the shared softmax so they
        # can NEVER be completely suppressed by the optimizer.
        k_tokens = self.k_self(x_norm)
        v_tokens = self.v_self(x_norm)
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)
        k_task_proj = self.k_task(h_task)
        v_task_proj = self.v_task(h_task)

        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_mh = reshape_heads(q_1, B, T)
        k_self_mh = reshape_heads(k_tokens, B, T)
        v_self_mh = reshape_heads(v_tokens, B, T)
        k_adap_mh = reshape_heads(k_adapter, B, K_a)
        v_adap_mh = reshape_heads(v_adapter, B, K_a)
        k_task_mh = reshape_heads(k_task_proj, B, K_t)
        v_task_mh = reshape_heads(v_task_proj, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_roped, k_self_roped = apply_rope(q_mh, k_self_mh, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adap_roped = apply_rope(k_adap_mh, k_adap_mh, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task_roped = apply_rope(k_task_mh, k_task_mh, cos_t, sin_t)

        # Shared softmax over self + adapter + task (vision always active)
        k_sat = torch.cat([k_self_roped, k_adap_roped, k_task_roped], dim=2)
        v_sat = torch.cat([v_self_mh, v_adap_mh, v_task_mh], dim=2)
        out_sat = self._branch_attention(q_roped, k_sat, v_sat, scale)
        attn_out = self.o_proj(out_sat)  # (B, T, C)

        # --- Depth branch (independent attention, gated, optional) ---
        if self.use_depth and h_d is not None:
            K_d = h_d.size(1)
            k_depth_proj = self.k_depth(h_d)
            v_depth_proj = self.v_depth(h_d)
            k_depth_mh = reshape_heads(k_depth_proj, B, K_d)
            v_depth_mh = reshape_heads(v_depth_proj, B, K_d)
            cos_d, sin_d = self.rope(seq_len=K_d, device=x.device, dtype=x.dtype)
            _, k_depth_roped = apply_rope(k_depth_mh, k_depth_mh, cos_d, sin_d)
            ratio_g_depth = torch.tanh(self.gating_depth)
            out_depth = self._branch_attention(q_roped, k_depth_roped, v_depth_mh, scale)
            attn_out = attn_out + ratio_g_depth * self.depth_out_ln(self.o_proj_depth(out_depth))

        # --- Force branch (independent attention, gated, optional) ---
        if self.use_force and h_f is not None:
            K_f = h_f.size(1)
            k_force_proj = self.k_force(h_f)
            v_force_proj = self.v_force(h_f)
            k_force_mh = reshape_heads(k_force_proj, B, K_f)
            v_force_mh = reshape_heads(v_force_proj, B, K_f)
            cos_f, sin_f = self.rope(seq_len=K_f, device=x.device, dtype=x.dtype)
            _, k_force_roped = apply_rope(k_force_mh, k_force_mh, cos_f, sin_f)
            ratio_g_force = torch.tanh(self.gating_force)
            out_force = self._branch_attention(q_roped, k_force_roped, v_force_mh, scale)
            attn_out = attn_out + ratio_g_force * self.force_out_ln(self.o_proj_force(out_force))

        # --- Residual 1: attention (post-attn LN prevents scale explosion) ---
        x = x + self.post_attn_ln(attn_out)

        # --- Residual 2: FFN with pre-norm ---
        x = x + self.ffn_act(self.ffn_linear(self.ffn_ln(x)))

        return x
