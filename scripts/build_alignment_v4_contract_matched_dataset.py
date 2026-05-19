#!/usr/bin/env python3
"""Build the contract-matched V4 dataset.

This is currently a thin wrapper around the short-horizon teacher converter,
with a different output name to keep the V4 pipeline explicit.
"""

from __future__ import annotations

import sys

from build_alignment_v4_short_horizon_teacher import main


if __name__ == "__main__":
    if "--output_name" not in sys.argv:
        sys.argv += ["--output_name", "alignment_v4_contract_matched_dataset.npz"]
    main()
