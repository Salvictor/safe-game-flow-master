"""Continuous trajectory representations."""

from .boundary_conditioned import (
    boundary_envelope,
    decode_boundary_residual,
    encode_boundary_residual,
    quintic_boundary_skeleton,
)
from .quintic import QuinticSegment, fit_quintic_segment
from .velocity_conditioned import (
    integrate_velocity_profile,
    project_velocity_boundary_constraints,
)

__all__ = [
    "QuinticSegment",
    "fit_quintic_segment",
    "boundary_envelope",
    "decode_boundary_residual",
    "encode_boundary_residual",
    "quintic_boundary_skeleton",
    "integrate_velocity_profile",
    "project_velocity_boundary_constraints",
]
