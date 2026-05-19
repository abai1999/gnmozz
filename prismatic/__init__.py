"""Prismatic package entry point.

The full VLM stack pulls in optional heavyweight dependencies (e.g. transformers,
hf hub, draccus CLI plumbing).  The local-proposal line only needs the lightweight
subpackages, so we keep top-level import side effects minimal here.
"""

__all__ = []
