import numpy as np
import pytest
import torch

from safe_game_flow.flow_matching.dataloader import TrajDataset
from safe_game_flow.flow_matching.model import FlowMatching1D, model_config_from_checkpoint
from safe_game_flow.flow_matching.normalization import (
    denormalize_trajectory,
    denormalize_velocity,
    normalize_trajectory,
    normalize_velocity,
)
from safe_game_flow.flow_matching.train import sample_ode


@pytest.mark.parametrize("channels", [2, 3])
def test_dataset_supports_arbitrary_channels(tmp_path, channels):
    rng = np.random.default_rng(7)
    data = rng.normal(size=(12, channels, 16)).astype(np.float32)
    path = tmp_path / f"traj_{channels}d.npy"
    np.save(path, data)

    dataset = TrajDataset(str(path), normalize=True)

    assert dataset.channels == channels
    assert dataset.H == 16
    assert dataset[0].shape == (channels, 16)
    assert dataset.mean.shape == (channels, 1)
    assert dataset.std.shape == (channels, 1)
    assert torch.allclose(dataset.data.mean(dim=(0, 2)), torch.zeros(channels), atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_position_and_velocity_normalization_round_trip(backend):
    data_np = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    velocity_np = np.linspace(-2, 2, 30, dtype=np.float32).reshape(2, 3, 5)
    mean_np = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    std_np = np.array([[2.0], [4.0], [5.0]], dtype=np.float32)

    if backend == "torch":
        data = torch.from_numpy(data_np)
        velocity = torch.from_numpy(velocity_np)
        mean = torch.from_numpy(mean_np)
        std = torch.from_numpy(std_np)
        assert torch.allclose(
            denormalize_trajectory(normalize_trajectory(data, mean, std), mean, std), data
        )
        assert torch.allclose(
            denormalize_velocity(normalize_velocity(velocity, std), std), velocity
        )
    else:
        np.testing.assert_allclose(
            denormalize_trajectory(normalize_trajectory(data_np, mean_np, std_np), mean_np, std_np),
            data_np,
        )
        np.testing.assert_allclose(
            denormalize_velocity(normalize_velocity(velocity_np, std_np), std_np), velocity_np
        )


@pytest.mark.parametrize("channels", [2, 3])
def test_model_preserves_channel_and_horizon_shape(channels):
    model = FlowMatching1D(
        in_channels=channels,
        hidden_channels=12,
        num_blocks=2,
        time_emb_dim=16,
        kernel_size=3,
    )
    x = torch.randn(4, channels, 19)
    t = torch.rand(4)

    output = model(x, t)

    assert output.shape == x.shape
    assert model.get_config()["in_channels"] == channels


def test_sample_ode_is_reproducible_and_uses_model_channels():
    model = FlowMatching1D(
        in_channels=3,
        hidden_channels=8,
        num_blocks=1,
        time_emb_dim=16,
    )
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    first = sample_ode(
        model, num_samples=2, H=11, ode_steps=2, device=torch.device("cpu"),
        generator=generator_a,
    )
    second = sample_ode(
        model, num_samples=2, H=11, ode_steps=2, device=torch.device("cpu"),
        generator=generator_b,
    )

    assert first.shape == (2, 3, 11)
    assert torch.equal(first, second)


def test_legacy_checkpoint_config_is_inferred_from_state_dict():
    model = FlowMatching1D(
        in_channels=3,
        hidden_channels=12,
        num_blocks=2,
        time_emb_dim=16,
        kernel_size=5,
    )

    config = model_config_from_checkpoint({"model": model.state_dict()})

    assert config == model.get_config()


def test_dataset_loads_and_normalizes_npz_conditions(tmp_path):
    rng = np.random.default_rng(11)
    trajectories = rng.normal(size=(10, 3, 12)).astype(np.float32)
    conditions = rng.normal(size=(10, 7)).astype(np.float32)
    trajectory_path = tmp_path / "trajectories.npy"
    condition_path = tmp_path / "conditions.npz"
    np.save(trajectory_path, trajectories)
    np.savez(condition_path, values=conditions, schema=np.asarray([f"c{i}" for i in range(7)]))

    dataset = TrajDataset(
        str(trajectory_path), normalize=True, condition_path=str(condition_path)
    )
    trajectory, condition = dataset[0]

    assert trajectory.shape == (3, 12)
    assert condition.shape == (7,)
    assert dataset.condition_dim == 7
    assert dataset.condition_schema == tuple(f"c{i}" for i in range(7))
    assert torch.allclose(dataset.condition_data.mean(dim=0), torch.zeros(7), atol=1e-6)


def test_conditional_model_requires_valid_condition_and_preserves_shape():
    model = FlowMatching1D(
        in_channels=3,
        hidden_channels=12,
        num_blocks=2,
        time_emb_dim=16,
        condition_dim=7,
    )
    x = torch.randn(4, 3, 15)
    t = torch.rand(4)
    condition = torch.randn(4, 7)

    output = model(x, t, condition)

    assert output.shape == x.shape
    assert model.get_config()["condition_dim"] == 7
    with pytest.raises(ValueError, match="requires condition"):
        model(x, t)
    with pytest.raises(ValueError, match="Expected condition"):
        model(x, t, torch.randn(4, 6))


def test_conditional_sample_ode_is_reproducible():
    model = FlowMatching1D(
        in_channels=3,
        hidden_channels=8,
        num_blocks=1,
        time_emb_dim=16,
        condition_dim=5,
    )
    condition = torch.linspace(-1.0, 1.0, 5)
    generator_a = torch.Generator().manual_seed(321)
    generator_b = torch.Generator().manual_seed(321)

    first = sample_ode(
        model,
        num_samples=3,
        H=9,
        ode_steps=2,
        device=torch.device("cpu"),
        generator=generator_a,
        condition=condition,
    )
    second = sample_ode(
        model,
        num_samples=3,
        H=9,
        ode_steps=2,
        device=torch.device("cpu"),
        generator=generator_b,
        condition=condition,
    )

    assert first.shape == (3, 3, 9)
    assert torch.equal(first, second)
