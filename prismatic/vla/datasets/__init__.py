"""Lightweight dataset namespace for proposal-oriented experiments.

Keep RLBench/transformers-heavy imports out of package import time so the local
proposal pipeline can run in the minimal training environment.
"""

from .depth_force_candidate_future_risk_dataset import DepthForceCandidateFutureRiskDataset
from .depth_force_local_proposal_dataset import DepthForceLocalProposalDataset
from .alignment_v3_direct_local_dataset import AlignmentV3DirectLocalDataset
from .alignment_v4_direct_local_dataset import AlignmentV4DirectLocalDataset
from .alignment_diffusion_dataset import AlignmentDiffusionDataset
from .alignment_tc_student_vnext_dataset import AlignmentTCStudentVNextDataset

__all__ = [
    "AlignmentDiffusionDataset",
    "AlignmentTCStudentVNextDataset",
    "AlignmentV3DirectLocalDataset",
    "AlignmentV4DirectLocalDataset",
    "DepthForceCandidateFutureRiskDataset",
    "DepthForceLocalProposalDataset",
]
