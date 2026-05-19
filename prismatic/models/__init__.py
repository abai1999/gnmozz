"""Model package entry point.

Keep this lightweight for proposal-only experiments.  The heavyweight loading and
materialization helpers are available from their direct submodules when the full
stack dependencies are installed.
"""

from .residual_encoders import BaseActionEncoder, DepthEncoderTiny, ForceEncoderTiny, ProprioEncoder, RGBEncoderTiny
from .alignment_diffusion_refiner import AlignmentDiffusionRefiner
from .alignment_tc_diffusion_refiner import AlignmentTargetEstimator, TargetConditionedAlignmentDiffusionRefiner
from .alignment_tc_student_vnext import AlignmentTCStudentVNext
from .alignment_v3_direct_local_controller import AlignmentV3DirectLocalController
from .alignment_v4_direct_local_controller import AlignmentV4DirectLocalController

__all__ = [
    "AlignmentDiffusionRefiner",
    "AlignmentTargetEstimator",
    "AlignmentTCStudentVNext",
    "AlignmentV3DirectLocalController",
    "AlignmentV4DirectLocalController",
    "BaseActionEncoder",
    "DepthEncoderTiny",
    "ForceEncoderTiny",
    "ProprioEncoder",
    "RGBEncoderTiny",
    "TargetConditionedAlignmentDiffusionRefiner",
]
