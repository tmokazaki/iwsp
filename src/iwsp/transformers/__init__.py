"""
Transparent Transformers via Interpretable Weight-Space Programming (iWSP)
==========================================================================
"""

from iwsp.transformers.architecture import (
    AttentionHeadWeights,
    MLPWeights,
    TransformerLayerWeights,
    TransparentTransformer,
)
from iwsp.transformers.algorithmic import (
    ALGORITHMIC_BENCHMARKS,
    build_addition_transformer,
    build_gcd_transformer,
    build_majority_transformer,
    build_max_transformer,
    build_parity_transformer,
    build_sorting_transformer,
    evaluate_algorithmic_task,
    get_algorithmic_offline_trajectories,
)
from iwsp.transformers.nlp import (
    NLP_BENCHMARKS,
    build_mrpc_transformer,
    build_snli_transformer,
    build_sst2_transformer,
    build_trec_transformer,
    evaluate_nlp_task,
    get_nlp_offline_trajectories,
)

__all__ = [
    "AttentionHeadWeights",
    "MLPWeights",
    "TransformerLayerWeights",
    "TransparentTransformer",
    "ALGORITHMIC_BENCHMARKS",
    "build_addition_transformer",
    "build_sorting_transformer",
    "build_parity_transformer",
    "build_gcd_transformer",
    "build_max_transformer",
    "build_majority_transformer",
    "evaluate_algorithmic_task",
    "get_algorithmic_offline_trajectories",
    "NLP_BENCHMARKS",
    "build_sst2_transformer",
    "build_trec_transformer",
    "build_mrpc_transformer",
    "build_snli_transformer",
    "evaluate_nlp_task",
    "get_nlp_offline_trajectories",
]
