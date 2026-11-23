"""
Transparent Radio Simulation & Array Beamforming Models via iWSP
==================================================================
"""

from iwsp.radio.beamforming import (
    AOA_ANGLES_DEG,
    NUM_ELEMENTS,
    PHASE_BINS,
    build_beamforming_aoa_transformer,
    evaluate_beamforming_task,
    generate_beamforming_dataset,
    get_beamforming_offline_trajectory,
    quantize_phase_tokens,
)
from iwsp.radio.propagation import (
    DEFAULT_WALLS,
    WallSegment,
    count_wall_crossings,
    evaluate_radio_propagation,
    extract_radio_propagation_features,
    generate_radio_propagation_dataset,
    get_radio_propagation_offline_trajectory,
    render_ascii_radio_heatmap,
    segments_intersect,
)

__all__ = [
    "AOA_ANGLES_DEG",
    "NUM_ELEMENTS",
    "PHASE_BINS",
    "build_beamforming_aoa_transformer",
    "evaluate_beamforming_task",
    "generate_beamforming_dataset",
    "get_beamforming_offline_trajectory",
    "quantize_phase_tokens",
    "DEFAULT_WALLS",
    "WallSegment",
    "count_wall_crossings",
    "evaluate_radio_propagation",
    "extract_radio_propagation_features",
    "generate_radio_propagation_dataset",
    "get_radio_propagation_offline_trajectory",
    "render_ascii_radio_heatmap",
    "segments_intersect",
]
