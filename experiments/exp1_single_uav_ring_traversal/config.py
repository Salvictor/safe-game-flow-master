"""Configuration for the single-UAV ring traversal dataset."""

from pathlib import Path

from safe_game_flow.data import RingDatasetConfig


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"

DEFAULT_NUM_TRAJECTORIES = 2000
DEFAULT_SEED = 42
DATASET_CONFIG = RingDatasetConfig()
