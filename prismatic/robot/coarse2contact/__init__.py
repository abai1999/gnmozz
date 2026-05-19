"""Coarse2Contact runtime scaffold.

The package is intentionally runtime-only and non-privileged: it consumes the
frozen planner action chunk, wrist depth, proprioception, and force/torque
signals, then emits bounded local corrections plus trace fields.
"""

from .runtime import (  # noqa: F401
    Coarse2ContactContactState,
    Coarse2ContactController,
    Coarse2ContactPhase,
    Coarse2ContactState,
    Coarse2ContactSupervisor,
    ContactStateEstimator,
    DepthLocalizerEstimate,
    DepthVisualAligner,
    ForceReflexController,
    RecoveryPhase,
    RecoveryPrimitiveBank,
    RecoveryStateMachine,
    VisualAlignmentEstimate,
)

__all__ = [
    "Coarse2ContactContactState",
    "Coarse2ContactController",
    "Coarse2ContactPhase",
    "Coarse2ContactState",
    "Coarse2ContactSupervisor",
    "ContactStateEstimator",
    "DepthLocalizerEstimate",
    "DepthVisualAligner",
    "ForceReflexController",
    "RecoveryPhase",
    "RecoveryPrimitiveBank",
    "RecoveryStateMachine",
    "VisualAlignmentEstimate",
]
