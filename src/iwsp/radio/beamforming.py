"""
Phased-Array Beamforming & Angle-of-Arrival (AoA) Transformer via iWSP
======================================================================

Implements an 8-element Uniform Linear Array (ULA, spacing d = lambda / 2)
Angle-of-Arrival (AoA) / optimal beam selection Transformer programmed directly
in weight space without gradient descent.

Physical Formulation:
- A narrowband plane wave arriving from angle theta in [-60, +60] deg induces
  progressive phase shift across antenna elements m = 0..7:
      phi_m = phi_0 + m * pi * sin(theta)
- Input tokens: `[q_0, q_1, ..., q_7, "="]` where `q_m in {0..35}` represents
  the 10-degree quantized baseband phase phi_m = 2*pi*q_m / 36.
- Layer 0 Attention (`AntennaElementRoutingHead_0..7`) routes each element's
  in-phase/quadrature `(I_m, Q_m) = (cos(phi_m), sin(phi_m))` into dedicated
  residual stream channels `dim[11 + 2*m : 11 + 2*m + 2]` at `"="`.
- Unembedding (`W_U`) encodes the complex-conjugate spatial steering vectors
  `(cos(m * pi * sin(theta_k)), sin(m * pi * sin(theta_k)))` for the 9 candidate
  beams, so output logits compute the exact Bartlett spatial matched filter:
      logit(theta_k) = sum_{m=0}^{7} cos(phi_m - m * pi * sin(theta_k)).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

from iwsp.math_backend import np
from iwsp.transformers.architecture import (
    AttentionHeadWeights,
    TransformerLayerWeights,
    TransparentTransformer,
)

AOA_ANGLES_DEG: List[int] = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
NUM_ELEMENTS: int = 8
PHASE_BINS: int = 36


def quantize_phase_tokens(
    angle_deg: float,
    phase_offset_deg: float = 0.0,
    noise_phases_deg: Sequence[float] = (),
    num_elements: int = NUM_ELEMENTS,
    phase_bins: int = PHASE_BINS,
) -> List[int]:
    """
    Simulate received quantized baseband phase tokens `[q_0, ..., q_{M-1}]` for an
    M-element ULA (d = lambda/2) receiving a plane wave from `angle_deg`.
    """
    theta_rad = math.radians(angle_deg)
    tokens: List[int] = []
    for m in range(num_elements):
        jitter = float(noise_phases_deg[m]) if m < len(noise_phases_deg) else 0.0
        phi_rad = (
            math.radians(phase_offset_deg + jitter)
            + float(m) * math.pi * math.sin(theta_rad)
        )
        # Wrap to [0, 2*pi) and quantize to nearest bin in 0..phase_bins-1
        phi_wrapped = phi_rad % (2.0 * math.pi)
        q = int(round(phi_wrapped * phase_bins / (2.0 * math.pi))) % phase_bins
        tokens.append(q)
    return tokens


def build_beamforming_aoa_transformer(
    angles_deg: Sequence[int] = AOA_ANGLES_DEG,
    num_elements: int = NUM_ELEMENTS,
    phase_bins: int = PHASE_BINS,
) -> TransparentTransformer:
    """
    Hand-programmed TransparentTransformer for 8-element ULA AoA / beam selection.

    Residual stream layout (`d_model = 11 + 2 * num_elements = 27`):
    - dim[0]: `baseband_I` (cos(phi))
    - dim[1]: `baseband_Q` (sin(phi))
    - dim[2..9]: `pos_0..pos_7` (one-hot antenna element positional tags)
    - dim[10]: `pos_query` (1.0 at final `"="` token)
    - dim[11 + 2*m]: `elem_m_I`
    - dim[11 + 2*m + 1]: `elem_m_Q`
    """
    vocab: List[Any] = list(range(phase_bins)) + ["="]
    d_model = 11 + 2 * num_elements
    dim_names = (
        ["baseband_I", "baseband_Q"]
        + [f"pos_{m}" for m in range(num_elements)]
        + ["pos_query"]
    )
    for m in range(num_elements):
        dim_names.extend([f"elem_{m}_I", f"elem_{m}_Q"])

    W_E = np.zeros((len(vocab), d_model), dtype=np.float64)
    for q in range(phase_bins):
        phi = 2.0 * math.pi * float(q) / float(phase_bins)
        W_E[q, 0] = math.cos(phi)
        W_E[q, 1] = math.sin(phi)

    W_pos = np.zeros((num_elements + 1, d_model), dtype=np.float64)
    for m in range(num_elements):
        W_pos[m, 2 + m] = 1.0
    W_pos[num_elements, 10] = 1.0

    heads: List[AttentionHeadWeights] = []
    for m in range(num_elements):
        W_Q = np.zeros((d_model, 1), dtype=np.float64)
        W_Q[10, 0] = 1.0
        W_K = np.zeros((d_model, 1), dtype=np.float64)
        W_K[2 + m, 0] = 1.0
        W_V = np.zeros((d_model, 2), dtype=np.float64)
        W_V[0, 0] = 1.0
        W_V[1, 1] = 1.0
        W_O = np.zeros((2, d_model), dtype=np.float64)
        W_O[0, 11 + 2 * m] = 1.0
        W_O[1, 11 + 2 * m + 1] = 1.0

        heads.append(
            AttentionHeadWeights(
                name=f"AntennaElementRoutingHead_{m}",
                role=f"At '=' (pos {num_elements}), copy antenna element {m} (I, Q) into dim[{11+2*m}:{11+2*m+2}]",
                W_Q=W_Q,
                W_K=W_K,
                W_V=W_V,
                W_O=W_O,
                causal=True,
                scale=50.0,
            )
        )

    layer0 = TransformerLayerWeights(
        name="SpatialArrayGatheringLayer",
        role="Routes all 8 antenna element (I, Q) phasors to query token '='",
        heads=heads,
        mlp=None,
    )

    num_beams = len(angles_deg)
    W_U = np.zeros((d_model, num_beams), dtype=np.float64)
    for k, theta_deg in enumerate(angles_deg):
        sin_theta = math.sin(math.radians(float(theta_deg)))
        for m in range(num_elements):
            steering_phase = float(m) * math.pi * sin_theta
            W_U[11 + 2 * m, k] = math.cos(steering_phase)
            W_U[11 + 2 * m + 1, k] = math.sin(steering_phase)

    return TransparentTransformer(
        name="iWSP_ULA_Beamforming_AoA_Transformer",
        description=(
            "1-Layer, 8-Head transparent Transformer performing phased-array Angle-of-Arrival (AoA) "
            "and optimal beam selection via explicit complex conjugate steering vectors in W_U."
        ),
        d_model=d_model,
        dim_names=dim_names,
        W_E=W_E,
        W_pos=W_pos,
        layers=[layer0],
        W_U=W_U,
        vocab=vocab,
        readout_mode="last_token",
    )


def generate_beamforming_dataset(
    angles_deg: Sequence[int] = AOA_ANGLES_DEG,
) -> List[Tuple[List[Any], int]]:
    """
    Generate evaluation dataset across all 9 AoA beam directions and multiple
    coherent carrier phase offsets and element phase perturbations.
    """
    rng = np.random.RandomState(2026)
    dataset: List[Tuple[List[Any], int]] = []
    phase_offsets = [-35.0, -20.0, 0.0, 20.0, 35.0]
    for class_idx, theta_deg in enumerate(angles_deg):
        for offset in phase_offsets:
            # Clean quantized plane wave
            q_clean = quantize_phase_tokens(theta_deg, phase_offset_deg=offset)
            dataset.append((q_clean + ["="], class_idx))
            # Perturbed plane wave with per-element phase noise (std 8 deg)
            noise = [float(rng.normal(0.0, 8.0)) for _ in range(NUM_ELEMENTS)]
            q_noisy = quantize_phase_tokens(
                theta_deg, phase_offset_deg=offset, noise_phases_deg=noise
            )
            dataset.append((q_noisy + ["="], class_idx))
    return dataset


def evaluate_beamforming_task(
    model_builder: Callable[[], TransparentTransformer] = build_beamforming_aoa_transformer,
) -> Tuple[float, Dict[str, Any], str, TransparentTransformer]:
    """Evaluate a weight-space programmed AoA / beamforming Transformer."""
    model = model_builder()
    dataset = generate_beamforming_dataset()
    correct = 0
    for tokens, target_idx in dataset:
        pred = int(model.predict(tokens))
        if pred == target_idx:
            correct += 1
    acc = correct / max(len(dataset), 1)
    pstats = model.parameter_stats()
    metrics = {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": len(dataset),
        "angles_deg": list(AOA_ANGLES_DEG),
        **pstats,
    }
    return acc, metrics, model.description, model


def get_beamforming_offline_trajectory() -> Dict[str, List[Tuple[str, str]]]:
    """
    Provide a 2-step iWSP scientific refinement trajectory for phased-array beamforming:
    - Iteration 1: Uniform phase pooling ignoring antenna element order (fails on oblique angles).
    - Iteration 2: Positional element routing heads + complex conjugate spatial steering vectors.
    """
    return {
        "beamforming_aoa": [
            (
                "Attempt 1: Pool (I, Q) uniformly across all 8 antenna elements without positional steering vectors (treats array as omnidirectional point sensor; only broadside 0 deg aligns).",
                """
def build_model():
    from iwsp.radio.beamforming import AOA_ANGLES_DEG, PHASE_BINS
    vocab = list(range(PHASE_BINS)) + ["="]
    d_model = 4
    W_E = np.zeros((len(vocab), d_model))
    for q in range(PHASE_BINS):
        phi = 2.0 * math.pi * q / PHASE_BINS
        W_E[q, 0] = math.cos(phi)
        W_E[q, 1] = math.sin(phi)
        W_E[q, 2] = 1.0
    W_E[vocab.index("="), 3] = 1.0
    W_Q = np.zeros((d_model, 1)); W_Q[3, 0] = 40.0
    W_K = np.zeros((d_model, 1)); W_K[2, 0] = 40.0
    W_V = np.zeros((d_model, 2)); W_V[0, 0] = 1.0; W_V[1, 1] = 1.0
    W_O = np.zeros((2, d_model)); W_O[0, 0] = 1.0; W_O[1, 1] = 1.0
    head = AttentionHeadWeights("UniformPoolHead", "Uniform average over elements without spatial phase compensation", W_Q, W_K, W_V, W_O, scale=1.0)
    W_U = np.zeros((d_model, len(AOA_ANGLES_DEG)))
    for k, ang in enumerate(AOA_ANGLES_DEG):
        W_U[0, k] = math.cos(math.radians(ang))
        W_U[1, k] = math.sin(math.radians(ang))
    return TransparentTransformer("Beamforming_v1", "Unsteered uniform pooling baseline", d_model, ["I", "Q", "is_elem", "is_q"], W_E, [TransformerLayerWeights("L0", "Pool", [head])], W_U, vocab=vocab)
""",
            ),
            (
                "Attempt 2: Program 8 positional element-routing heads to gather (I_m, Q_m) and hardcode conjugate spatial steering vectors (cos(m*pi*sin(theta_k)), sin(m*pi*sin(theta_k))) in W_U.",
                """
def build_model():
    from iwsp.radio.beamforming import build_beamforming_aoa_transformer
    return build_beamforming_aoa_transformer()
""",
            ),
        ]
    }
