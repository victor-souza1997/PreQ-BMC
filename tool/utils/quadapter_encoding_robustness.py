"""Compatibility wrapper for the legacy robustness encoding module."""

from synthesis.quadapter import (
    GPEncoding,
    LayerEncoding,
    QuadapterConfig,
    QuadapterRobustnessSynthesizer,
    SynthesisResult,
)

__all__ = [
    "GPEncoding",
    "LayerEncoding",
    "QuadapterConfig",
    "QuadapterRobustnessSynthesizer",
    "SynthesisResult",
]
