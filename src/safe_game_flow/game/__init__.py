"""Game-theoretic interaction models for multi-UAV experiments."""

from .two_uav_ring import (
    GameSchedule,
    HOCBFProjection,
    select_pure_nash_schedule,
    project_inter_uav_hocbf,
)
from .two_uav_expert import (
    CONDITION_SCHEMA,
    JointRollout,
    QuinticReference,
    RingGameScene,
    condition_vector,
    make_reference,
    nominal_crossing_times,
    reference_positions,
    resample_joint_positions,
    resample_reference_positions,
    sample_scene,
    scheduled_references,
    simulate_expert,
    simulate_joint_reference,
)

__all__ = [
    "GameSchedule",
    "HOCBFProjection",
    "select_pure_nash_schedule",
    "project_inter_uav_hocbf",
    "CONDITION_SCHEMA",
    "JointRollout",
    "QuinticReference",
    "RingGameScene",
    "condition_vector",
    "make_reference",
    "nominal_crossing_times",
    "reference_positions",
    "resample_joint_positions",
    "resample_reference_positions",
    "sample_scene",
    "scheduled_references",
    "simulate_expert",
    "simulate_joint_reference",
]
