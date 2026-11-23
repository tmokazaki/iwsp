"""
iwsp: Interpretable Weight-Space Programming
============================================

A comprehensive Python package reproducing the core methodology and experiments of:
"Interpretable weight-space programming builds transparent transformers and brain encoding models"
(Mehta et al., arXiv:2611.18114v1).

Core modules:
- `iwsp.core`: The iWSP scientific agent loop, sandboxed code execution, and LLM/offline backends.
- `iwsp.transformers`: Glass-box Transformer engine, mechanistic audit reports, and weight-space programs
  for algorithmic benchmarks (Addition, Sorting, Parity, GCD, etc.) and NLP classification (SST-2, TREC, MRPC, SNLI).
- `iwsp.brain`: Transparent brain encoding models with cross-validated Ridge regression readouts across
  Visual Cortex (7T fMRI / NSD), Language fMRI (surfacing scan-edge timing confounds), and Auditory ECoG.
- `iwsp.radio`: Transparent RF/wireless simulation models (8-element ULA beamforming/AoA Transformer
  and 2D radio wave propagation encoding with wall ray-casting, two-ray reflection, and anomaly discovery).
"""

from iwsp.core import (
    IWSPHarvester,
    IWSPLoop,
    IWSPTrial,
    LLMBackend,
    OfflineScientistBackend,
    APIScientistBackend,
    SandboxExecutor,
)
from iwsp.transformers.architecture import (
    TransparentTransformer,
    TransformerLayerWeights,
    AttentionHeadWeights,
    MLPWeights,
)
from iwsp.brain.ridge import RidgeEncodingReadout, BrainEncodingEvaluator
from iwsp.radio import (
    build_beamforming_aoa_transformer,
    evaluate_beamforming_task,
    evaluate_radio_propagation,
    render_ascii_radio_heatmap,
)

__version__ = "0.1.0"
__all__ = [
    "IWSPHarvester",
    "IWSPLoop",
    "IWSPTrial",
    "LLMBackend",
    "OfflineScientistBackend",
    "APIScientistBackend",
    "SandboxExecutor",
    "TransparentTransformer",
    "TransformerLayerWeights",
    "AttentionHeadWeights",
    "MLPWeights",
    "RidgeEncodingReadout",
    "BrainEncodingEvaluator",
    "build_beamforming_aoa_transformer",
    "evaluate_beamforming_task",
    "evaluate_radio_propagation",
    "render_ascii_radio_heatmap",
]
