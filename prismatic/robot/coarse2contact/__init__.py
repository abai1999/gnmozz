"""Coarse2Contact runtime scaffold.

The package is intentionally runtime-only and non-privileged: it consumes the
frozen planner action chunk, wrist depth, proprioception, and force/torque
signals, then emits bounded local corrections plus trace fields.
"""

from .supervisor import (
    Coarse2ContactContactState,
    Coarse2ContactPhase,
    Coarse2ContactSupervisor,
    ContactStateEstimator,
    DepthVisualAligner,
    ForceReflexController,
    RecoveryPrimitiveBank,
    VisualAlignmentEstimate,
)

__all__ = [
    "Coarse2ContactContactState",
    "Coarse2ContactPhase",
    "Coarse2ContactSupervisor",
    "ContactStateEstimator",
    "DepthVisualAligner",
    "ForceReflexController",
    "RecoveryPrimitiveBank",
    "VisualAlignmentEstimate",
]
