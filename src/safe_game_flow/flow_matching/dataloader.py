
import numpy as np
import torch
from torch.utils.data import Dataset

from safe_game_flow.flow_matching.normalization import normalize_trajectory


# ----------------------------
# 数据集与标准化
# ----------------------------
class TrajDataset(Dataset):
    def __init__(
        self,
        npy_path: str,
        normalize: bool = True,
        condition_path: str | None = None,
        normalize_conditions: bool = True,
        trajectory_mean: np.ndarray | torch.Tensor | None = None,
        trajectory_std: np.ndarray | torch.Tensor | None = None,
        condition_mean: np.ndarray | torch.Tensor | None = None,
        condition_std: np.ndarray | torch.Tensor | None = None,
    ):
        arr = np.load(npy_path)  # shape: (N, C, H)
        if arr.ndim != 3:
            raise ValueError(f"Expected trajectory data with shape (N,C,H), got {arr.shape}")
        if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
            raise ValueError(f"Trajectory dimensions must be non-zero, got {arr.shape}")
        self.channels = arr.shape[1]
        self.H = arr.shape[-1]
        self.data = torch.from_numpy(arr).float()  # (N,C,H)

        # 计算每个通道的全局均值/方差 (在N和H维上)
        if normalize:
            if (trajectory_mean is None) != (trajectory_std is None):
                raise ValueError("trajectory_mean and trajectory_std must be provided together")
            if trajectory_mean is None:
                mean = self.data.mean(dim=(0, 2), keepdim=False)  # (C,)
                std = self.data.std(dim=(0, 2), keepdim=False).clamp_min(1e-6)  # (C,)
            else:
                mean = torch.as_tensor(trajectory_mean, dtype=torch.float32).detach().cpu().reshape(-1)
                std = torch.as_tensor(trajectory_std, dtype=torch.float32).detach().cpu().reshape(-1)
                if mean.shape != (self.channels,) or std.shape != (self.channels,):
                    raise ValueError(
                        "Trajectory statistics must have one value per channel: "
                        f"mean={tuple(mean.shape)}, std={tuple(std.shape)}, C={self.channels}"
                    )
                if torch.any(std <= 0):
                    raise ValueError("Trajectory standard deviations must be positive")

            self.mean = mean.view(self.channels, 1).clone()
            self.std = std.view(self.channels, 1).clone()
            self.data = normalize_trajectory(self.data, self.mean, self.std)
        else:
            self.mean = torch.zeros(self.channels, 1)
            self.std = torch.ones(self.channels, 1)

        self.condition_data: torch.Tensor | None = None
        self.condition_mean: torch.Tensor | None = None
        self.condition_std: torch.Tensor | None = None
        self.condition_schema: tuple[str, ...] | None = None
        self.condition_dim = 0
        if condition_path is not None:
            loaded = np.load(condition_path)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                try:
                    if "values" not in loaded:
                        raise ValueError(
                            f"Condition archive {condition_path} must contain a 'values' array"
                        )
                    conditions = loaded["values"]
                    if "schema" in loaded:
                        self.condition_schema = tuple(str(item) for item in loaded["schema"])
                finally:
                    loaded.close()
            else:
                conditions = loaded

            if conditions.ndim != 2:
                raise ValueError(
                    f"Expected condition data with shape (N,D), got {conditions.shape}"
                )
            if conditions.shape[0] != self.data.shape[0]:
                raise ValueError(
                    "Trajectory/condition sample count mismatch: "
                    f"{self.data.shape[0]} trajectories and {conditions.shape[0]} conditions"
                )
            if conditions.shape[1] == 0:
                raise ValueError("Condition dimension must be non-zero")
            if not np.isfinite(conditions).all():
                raise ValueError("Condition data contains NaN or infinite values")

            condition_tensor = torch.from_numpy(np.asarray(conditions)).float()
            self.condition_dim = int(condition_tensor.shape[1])
            if self.condition_schema is not None and len(self.condition_schema) != self.condition_dim:
                raise ValueError(
                    f"Condition schema has {len(self.condition_schema)} names for D={self.condition_dim}"
                )
            if normalize_conditions:
                if (condition_mean is None) != (condition_std is None):
                    raise ValueError("condition_mean and condition_std must be provided together")
                if condition_mean is None:
                    used_condition_mean = condition_tensor.mean(dim=0)
                    used_condition_std = condition_tensor.std(dim=0).clamp_min(1e-6)
                else:
                    used_condition_mean = torch.as_tensor(
                        condition_mean, dtype=torch.float32
                    ).detach().cpu().reshape(-1)
                    used_condition_std = torch.as_tensor(
                        condition_std, dtype=torch.float32
                    ).detach().cpu().reshape(-1)
                    expected_shape = (self.condition_dim,)
                    if (
                        used_condition_mean.shape != expected_shape
                        or used_condition_std.shape != expected_shape
                    ):
                        raise ValueError(
                            "Condition statistics must have shape (D,): "
                            f"mean={tuple(used_condition_mean.shape)}, "
                            f"std={tuple(used_condition_std.shape)}, D={self.condition_dim}"
                        )
                    if torch.any(used_condition_std <= 0):
                        raise ValueError("Condition standard deviations must be positive")
                condition_tensor = (
                    condition_tensor - used_condition_mean
                ) / used_condition_std
            else:
                used_condition_mean = torch.zeros(self.condition_dim)
                used_condition_std = torch.ones(self.condition_dim)
            self.condition_data = condition_tensor
            self.condition_mean = used_condition_mean
            self.condition_std = used_condition_std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.condition_data is None:
            return self.data[idx]
        return self.data[idx], self.condition_data[idx]
