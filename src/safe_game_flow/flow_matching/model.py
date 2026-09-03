import math

import torch
import torch.nn as nn


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest valid GroupNorm group count up to ``maximum``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def model_config_from_checkpoint(checkpoint: dict) -> dict[str, int]:
    """Read model configuration from new or legacy checkpoints.

    Checkpoints written before format version 2 did not store constructor
    arguments. Their configuration is inferred from state-dict tensor shapes.
    """
    if "model_config" in checkpoint:
        return dict(checkpoint["model_config"])

    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")

    required = [
        "in_proj.weight",
        "time_emb.mlp.0.weight",
        "blocks.0.conv1.weight",
    ]
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Cannot infer legacy model config; missing keys: {missing}")

    block_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    if not block_indices:
        raise ValueError("Cannot infer num_blocks from legacy checkpoint")

    return {
        "in_channels": int(state["in_proj.weight"].shape[1]),
        "hidden_channels": int(state["in_proj.weight"].shape[0]),
        "num_blocks": max(block_indices) + 1,
        "time_emb_dim": int(state["time_emb.mlp.0.weight"].shape[0]),
        "kernel_size": int(state["blocks.0.conv1.weight"].shape[-1]),
        "condition_dim": int(state["condition_emb.0.weight"].shape[1])
        if "condition_emb.0.weight" in state
        else 0,
    }


# ----------------------------
# 时间嵌入和网络
# ----------------------------
class FourierTimeEmbedding(nn.Module):
    def __init__(self, num_frequencies: int = 16, out_dim: int = 128):
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be at least 1")
        self.num_frequencies = num_frequencies
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_frequencies, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0,1]
        # 使用 2^k * 2pi * t 的傅里叶特征
        device = t.device
        k = torch.arange(self.num_frequencies, device=device).float()
        freqs = (2.0 ** k) * 2.0 * math.pi  # (F,)
        # (B, F)
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 2F)
        return self.mlp(emb)  # (B, out_dim)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, t_dim: int, groups: int = 8, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        group_count = _group_count(channels, groups)
        self.norm1 = nn.GroupNorm(num_groups=group_count, num_channels=channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups=group_count, num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(t_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H), t_emb: (B,t_dim)
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        # 注入时间偏置
        tb = self.time_proj(t_emb).unsqueeze(-1)  # (B,C,1)
        h = h + tb
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class FlowMatching1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 256,
        num_blocks: int = 6,
        time_emb_dim: int = 128,
        kernel_size: int = 3,
        condition_dim: int = 0,
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be at least 1")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be at least 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        if time_emb_dim < 8:
            raise ValueError("time_emb_dim must be at least 8")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if condition_dim < 0:
            raise ValueError("condition_dim must be non-negative")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.time_emb_dim = time_emb_dim
        self.kernel_size = kernel_size
        self.condition_dim = condition_dim
        self.time_emb = FourierTimeEmbedding(
            num_frequencies=time_emb_dim // 8, out_dim=time_emb_dim
        )
        self.condition_emb = (
            nn.Sequential(
                nn.Linear(condition_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            if condition_dim > 0
            else None
        )
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            ResBlock1D(hidden_channels, t_dim=time_emb_dim, kernel_size=kernel_size)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.GroupNorm(
            num_groups=_group_count(hidden_channels), num_channels=hidden_channels
        )
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

        # 小初始化，帮助稳定训练
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B,C,H); t: (B,) in [0,1]; condition: optional (B,D)
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,C,H), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"Expected t with shape ({x.shape[0]},), got {tuple(t.shape)}")
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        if self.condition_dim > 0:
            if condition is None:
                raise ValueError(
                    f"This model requires condition with shape (B,{self.condition_dim})"
                )
            if condition.ndim != 2 or condition.shape != (x.shape[0], self.condition_dim):
                raise ValueError(
                    f"Expected condition with shape ({x.shape[0]},{self.condition_dim}), "
                    f"got {tuple(condition.shape)}"
                )
            t_emb = t_emb + self.condition_emb(condition)
        elif condition is not None:
            raise ValueError("An unconditional model does not accept condition input")
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = self.out_norm(h)
        h = self.out_act(h)
        v = self.out_proj(h)  # (B,C,H)
        return v

    def get_config(self) -> dict[str, int]:
        """Return constructor arguments required to recreate this model."""
        return {
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "num_blocks": self.num_blocks,
            "time_emb_dim": self.time_emb_dim,
            "kernel_size": self.kernel_size,
            "condition_dim": self.condition_dim,
        }
