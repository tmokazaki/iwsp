"""
Neuroscience Brain Encoding Modalities via iWSP (arXiv:2611.18114)
==================================================================

Reproduces the three scientific brain-encoding domains investigated in the paper:

1. **Visual Cortex (7T fMRI / Natural Scenes Dataset - NSD)**:
   - Evaluates zero-shot interpretable visual features on early visual cortex (`V1_Early`)
     versus higher ventral visual cortex (`V4_IT_Ventral`).
   - Reproduces the paper's key neuroscientific dissociation:
     - `V1_Early` is driven by local oriented spatial frequency / Sobel edge energy and
       center-surround luminance contrast.
     - `V4_IT_Ventral` is driven by mid/high-level warm-cool color opponency, spatial
       envelope symmetry, and figure-ground radial centrality.

2. **Language fMRI (Story Listening fMRI)**:
   - Maps timestamped narrative clauses `(text, tr_index, total_trs)` to language-network
     voxels (`Temp_Semantic`, `IFG_Syntactic`) and scan-sensitive voxels (`Scan_Boundary_ROI`).
   - Reproduces one of the paper's headline discoveries: **surfacing hidden data confounds**.
     When the iWSP agent engineers features to maximize fMRI validation $r$, it discovers that
     an explicit **scan-edge timing / story-position indicator** (`is_scan_edge`, firing at the
     start and end of fMRI runs due to hemodynamic onset/offset transients) explains massive
     variance in `Scan_Boundary_ROI` that standard black-box LLM embeddings silently absorb!

3. **Auditory ECoG (Electrocorticography during Natural Sound / Speech Listening)**:
   - Maps multi-band time-frequency spectrograms `(freq_bins x time_steps)` to auditory cortex
     electrodes (`HG_Primary_A1`, `STG_Onset_Transient`, `STG_Harmonic_Pitch`).
   - Systematically discovers interpretable acoustic features: broadband RMS envelope,
     positive half-wave rectified spectral flux (acoustic onset detector), and spectral
     centroid / high-to-low frequency harmonic ratio.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Sequence, Tuple

from iwsp.brain.ridge import BrainEncodingEvaluator, RidgeEncodingReadout
from iwsp.math_backend import np


# ===========================================================================
# 1. Visual Cortex (NSD 7T fMRI: Early V1 vs Higher Ventral V4/IT)
# ===========================================================================

def generate_visual_cortex_dataset(
    n_train: int = 80, n_test: int = 40, grid_size: int = 8, seed: int = 42
) -> Tuple[List[Any], Any, List[Any], Any, List[str]]:
    """
    Generate synthetic NSD-style RGB visual stimuli `(grid_size, grid_size, 3)` and
    2 regional 7T fMRI responses (`V1_Early`, `V4_IT_Ventral`) with realistic SNR.
    """
    rng = np.random.RandomState(seed)
    total = n_train + n_test
    stimuli: List[Any] = []
    y_rows: List[List[float]] = []

    for _ in range(total):
        # Create structured natural-like 8x8x3 RGB patch with grating edges, center object, and color
        img = rng.uniform(0.1, 0.9, size=(grid_size, grid_size, 3))
        # Inject oriented gratings or central blobs randomly
        h_freq = rng.uniform(0.0, 1.5)
        v_freq = rng.uniform(0.0, 1.5)
        warm_bias = rng.uniform(-0.5, 0.5)
        center_obj = rng.uniform(0.0, 1.2)

        img_list = img.tolist()
        mid = (grid_size - 1) / 2.0
        for r in range(grid_size):
            for c in range(grid_size):
                grating = 0.35 * h_freq * (1.0 if r % 2 == 0 else -1.0) + 0.35 * v_freq * (1.0 if c % 2 == 0 else -1.0)
                dist = math.sqrt((r - mid) ** 2 + (c - mid) ** 2)
                radial = center_obj * math.exp(-(dist ** 2) / 4.0)
                red = min(max(img_list[r][c][0] + grating + radial + warm_bias, 0.0), 2.0)
                green = min(max(img_list[r][c][1] + grating + 0.8 * radial, 0.0), 2.0)
                blue = min(max(img_list[r][c][2] + grating + 0.3 * radial - warm_bias, 0.0), 2.0)
                img_list[r][c] = [red, green, blue]

        stim = np.array(img_list, dtype=np.float64)
        stimuli.append(stim)

    # Extract ground-truth cortical feature generators to synthesize V1 and V4/IT BOLD signals
    X_gt, _ = extract_visual_cortex_features(stimuli)
    X_gt_list = X_gt.tolist()
    # Features:
    # 0: mean_luminance, 1: rms_contrast, 2: horiz_edge_energy, 3: vert_edge_energy,
    # 4: red_blue_warmth, 5: center_surround_saliency, 6: spatial_symmetry
    for row in X_gt_list:
        lum, contrast, h_edge, v_edge, warmth, center_sal, sym = row
        # V1_Early responds strongly to oriented edge energy (horiz + vert) and RMS contrast
        v1_resp = (
            1.8 * h_edge
            + 1.7 * v_edge
            + 1.1 * contrast
            + 0.2 * lum
            + float(rng.normal(0.0, 0.08))
        )
        # V4_IT_Ventral responds strongly to color opponency (warmth), center-surround figure saliency, and symmetry
        v4_it_resp = (
            1.9 * warmth
            + 1.8 * center_sal
            + 1.2 * sym
            + 0.2 * h_edge
            + float(rng.normal(0.0, 0.08))
        )
        y_rows.append([v1_resp, v4_it_resp])

    Y = np.array(y_rows, dtype=np.float64)
    region_names = ["V1_Early", "V4_IT_Ventral"]
    return (
        stimuli[:n_train],
        Y[:n_train],
        stimuli[n_train:],
        Y[n_train:],
        region_names,
    )


def extract_visual_cortex_features(stimuli: Sequence[Any]) -> Tuple[Any, List[str]]:
    """
    Hand-programmed interpretable visual cortex feature bank discovered by iWSP:
    1. `mean_luminance`: Global average intensity across the visual field.
    2. `rms_contrast`: Standard deviation of pixel luminance (early retinal/LGN gain).
    3. `horiz_edge_energy`: Mean absolute vertical derivative $|I_{r+1,c} - I_{r,c}|$ (V1 horizontal orientation columns).
    4. `vert_edge_energy`: Mean absolute horizontal derivative $|I_{r,c+1} - I_{r,c}|$ (V1 vertical orientation columns).
    5. `red_blue_warmth`: Warm-cool chromatic opponency $\text{mean}(R - B)$ (V4 color-selective globs).
    6. `center_surround_saliency`: Foveal center vs peripheral surround luminance ratio (V4/IT figure-ground selectivity).
    7. `spatial_symmetry`: Bilateral horizontal mirror similarity (higher ventral object selectivity).
    """
    feature_names = [
        "mean_luminance",
        "rms_contrast",
        "horiz_edge_energy",
        "vert_edge_energy",
        "red_blue_warmth",
        "center_surround_saliency",
        "spatial_symmetry",
    ]
    rows: List[List[float]] = []

    for stim in stimuli:
        img = stim.tolist() if hasattr(stim, "tolist") else stim
        H = len(img)
        W = len(img[0])
        lum_grid = [
            [(img[r][c][0] + img[r][c][1] + img[r][c][2]) / 3.0 for c in range(W)]
            for r in range(H)
        ]
        flat_lum = [lum_grid[r][c] for r in range(H) for c in range(W)]
        mean_lum = sum(flat_lum) / len(flat_lum)
        rms_contrast = math.sqrt(sum((x - mean_lum) ** 2 for x in flat_lum) / len(flat_lum))

        # Horizontal & vertical edge energy (V1 simple/complex cell Gabor proxy)
        h_diffs = [
            abs(lum_grid[r + 1][c] - lum_grid[r][c])
            for r in range(H - 1)
            for c in range(W)
        ]
        v_diffs = [
            abs(lum_grid[r][c + 1] - lum_grid[r][c])
            for r in range(H)
            for c in range(W - 1)
        ]
        horiz_edge = sum(h_diffs) / max(len(h_diffs), 1)
        vert_edge = sum(v_diffs) / max(len(v_diffs), 1)

        # Chromatic warm-cool opponency (R - B)
        rb_diffs = [img[r][c][0] - img[r][c][2] for r in range(H) for c in range(W)]
        warmth = sum(rb_diffs) / len(rb_diffs)

        # Foveal center-surround contrast (inner 4x4 vs outer ring)
        c_start, c_end = H // 4, H - (H // 4)
        center_vals: List[float] = []
        surround_vals: List[float] = []
        for r in range(H):
            for c in range(W):
                if c_start <= r < c_end and c_start <= c < c_end:
                    center_vals.append(lum_grid[r][c])
                else:
                    surround_vals.append(lum_grid[r][c])
        center_sal = (sum(center_vals) / max(len(center_vals), 1)) - (
            sum(surround_vals) / max(len(surround_vals), 1)
        )

        # Bilateral horizontal mirror symmetry
        sym_diffs = [
            abs(lum_grid[r][c] - lum_grid[r][W - 1 - c])
            for r in range(H)
            for c in range(W // 2)
        ]
        symmetry = -sum(sym_diffs) / max(len(sym_diffs), 1)

        rows.append([
            mean_lum,
            rms_contrast,
            horiz_edge,
            vert_edge,
            warmth,
            center_sal,
            symmetry,
        ])

    return np.array(rows, dtype=np.float64), feature_names


# ===========================================================================
# 2. Language fMRI (Story Listening + Scan-Edge Timing Confound Discovery)
# ===========================================================================

def generate_language_fmri_dataset(
    n_train: int = 72, n_test: int = 36, run_length: int = 18, seed: int = 101
) -> Tuple[List[Dict[str, Any]], Any, List[Dict[str, Any]], Any, List[str]]:
    """
    Generate narrative story-listening fMRI stimuli consisting of timestamped TR segments:
    `{"text": str, "tr_in_run": int, "run_length": int}` across multiple fMRI scanner runs,
    along with 3 voxel regions:
    - `Temp_Semantic`: Left anterior/posterior temporal cortex (responds to social/animate & spatial content)
    - `IFG_Syntactic`: Left inferior frontal gyrus (responds to clause complexity / word rate & subordination)
    - `Scan_Boundary_ROI`: Voxels heavily affected by scan-edge timing confounds (first/last 2 TRs of a scan run)
    """
    rng = np.random.RandomState(seed)
    social_templates = [
        "my mother told her friend that the family doctor smiled warmly",
        "the boy and his sister spoke with their teacher about the child",
        "she hugged her father while the crowd cheered for the singer",
        "our neighbor met the young woman and her brother at dinner",
    ]
    spatial_templates = [
        "we walked down the narrow street toward the brick building on the corner",
        "inside the dark room a wooden table stood beside the high window",
        "across the wide river the mountain road wound through the forest",
        "he drove past the station and parked near the bridge entrance",
    ]
    complex_clauses = [
        "although because the regulation that was proposed yesterday seemed simple , however if the committee objects",
        "while whenever the hypothesis which we tested fails , nevertheless unless the measurement is repeated",
        "whereas if the mechanism that governs the circuit is nonlinear , therefore although the approximation holds",
    ]
    neutral_fillers = [
        "it was quiet for a moment then things continued as usual",
        "nothing much happened during the middle part of the afternoon",
        "there were several items left on the list to check later",
    ]

    total = n_train + n_test
    stimuli: List[Dict[str, Any]] = []
    for idx in range(total):
        tr_in_run = idx % run_length
        bucket = idx % 4
        if bucket == 0:
            base = social_templates[idx % len(social_templates)]
        elif bucket == 1:
            base = spatial_templates[idx % len(spatial_templates)]
        elif bucket == 2:
            base = complex_clauses[idx % len(complex_clauses)]
        else:
            base = neutral_fillers[idx % len(neutral_fillers)]
        stimuli.append({
            "text": base,
            "tr_in_run": tr_in_run,
            "run_length": run_length,
        })

    X_gt, _ = extract_language_fmri_features(stimuli)
    X_gt_list = X_gt.tolist()
    y_rows: List[List[float]] = []
    # Features:
    # 0: word_count, 1: mean_word_len, 2: social_animate_density,
    # 3: spatial_scene_density, 4: syntactic_subordination_density,
    # 5: scan_edge_transient_confound, 6: relative_run_progress
    for row in X_gt_list:
        w_cnt, w_len, social, spatial, syntax, scan_edge, run_prog = row
        temp_sem = 2.0 * social + 1.4 * spatial + 0.3 * w_cnt + float(rng.normal(0.0, 0.1))
        ifg_syn = 2.2 * syntax + 0.9 * w_len + 0.4 * w_cnt + float(rng.normal(0.0, 0.1))
        # Scan_Boundary_ROI is dominated by the fMRI scan-edge transient confound!
        scan_roi = 2.8 * scan_edge - 0.5 * run_prog + 0.15 * w_cnt + float(rng.normal(0.0, 0.1))
        y_rows.append([temp_sem, ifg_syn, scan_roi])

    Y = np.array(y_rows, dtype=np.float64)
    region_names = ["Temp_Semantic", "IFG_Syntactic", "Scan_Boundary_ROI"]
    return (
        stimuli[:n_train],
        Y[:n_train],
        stimuli[n_train:],
        Y[n_train:],
        region_names,
    )


def extract_language_fmri_features(stimuli: Sequence[Dict[str, Any]]) -> Tuple[Any, List[str]]:
    """
    Hand-programmed interpretable Language fMRI feature set discovered by iWSP,
    explicitly isolating lexical/syntactic brain computations from the **fMRI scan-edge
    timing confound** highlighted in arXiv:2611.18114:
    1. `word_rate`: Token count per TR segment (auditory/lexical drive).
    2. `mean_word_length`: Average character length per word (orthographic/lexical complexity).
    3. `social_animate_density`: Rate of human/kinship/social agent nouns & pronouns.
    4. `spatial_scene_density`: Rate of physical place/navigation/topographic nouns & prepositions.
    5. `syntactic_subordination`: Rate of subordinating conjunctions & relative pronouns.
    6. `scan_edge_confound`: Exponential decay indicator active during the first 2 and last 2 TRs
       of an fMRI run ($\exp(-t/1.5) + \exp(-(T-1-t)/1.5)$), surfacing scanner boundary artifacts.
    7. `run_drift_position`: Normalized linear position $t / (T - 1)$ within the fMRI scan run.
    """
    social_words = {
        "mother", "father", "friend", "family", "doctor", "boy", "sister",
        "teacher", "child", "she", "her", "his", "crowd", "singer", "neighbor",
        "woman", "brother", "he", "my", "our", "their",
    }
    spatial_words = {
        "street", "building", "corner", "inside", "room", "table", "window",
        "across", "river", "mountain", "road", "forest", "station", "bridge",
        "entrance", "down", "toward", "beside", "near", "past",
    }
    syntax_words = {
        "although", "because", "that", "however", "if", "while", "whenever",
        "which", "nevertheless", "unless", "whereas", "therefore",
    }

    feature_names = [
        "word_rate",
        "mean_word_length",
        "social_animate_density",
        "spatial_scene_density",
        "syntactic_subordination",
        "scan_edge_confound",
        "run_drift_position",
    ]
    rows: List[List[float]] = []

    for item in stimuli:
        text = item["text"].lower()
        tr = float(item.get("tr_in_run", 5))
        run_len = float(max(item.get("run_length", 18), 2))
        words = re.findall(r"[a-z']+", text)
        n_w = max(len(words), 1)

        word_rate = float(len(words))
        mean_len = sum(len(w) for w in words) / n_w
        social_dens = sum(1.0 for w in words if w in social_words) / n_w
        spatial_dens = sum(1.0 for w in words if w in spatial_words) / n_w
        syntax_dens = sum(1.0 for w in words if w in syntax_words) / n_w

        # Scan-edge timing confound feature (exponential boundary indicator at start/end of scan)
        dist_start = tr
        dist_end = (run_len - 1.0) - tr
        scan_edge = math.exp(-dist_start / 1.2) + math.exp(-dist_end / 1.2)
        run_pos = tr / (run_len - 1.0)

        rows.append([
            word_rate,
            mean_len,
            social_dens,
            spatial_dens,
            syntax_dens,
            scan_edge,
            run_pos,
        ])

    return np.array(rows, dtype=np.float64), feature_names


# ===========================================================================
# 3. Auditory ECoG (Spectrogram -> Interpretable Acoustic Feature Discovery)
# ===========================================================================

def generate_auditory_ecog_dataset(
    n_train: int = 80,
    n_test: int = 40,
    n_freq: int = 8,
    n_time: int = 12,
    seed: int = 2026,
) -> Tuple[List[Any], Any, List[Any], Any, List[str]]:
    """
    Generate synthetic auditory spectrograms `(n_freq, n_time)` and intracranial ECoG
    high-gamma neural responses across 3 auditory cortical sites:
    - `HG_Primary_A1`: Heschl's gyrus / primary auditory cortex (tracks broadband acoustic power & spectral centroid)
    - `STG_Onset_Transient`: Superior temporal gyrus onset zone (tracks half-wave rectified positive spectral flux)
    - `STG_Harmonic_Pitch`: Pitch/voicing selective STG electrodes (tracks harmonic peak-to-floor ratio & low-mid periodicity)
    """
    rng = np.random.RandomState(seed)
    total = n_train + n_test
    stimuli: List[Any] = []

    for _ in range(total):
        base_energy = rng.uniform(0.1, 1.2)
        onset_burst = rng.uniform(0.0, 2.0)
        onset_t = int(rng.randint(2, n_time - 2))
        harmonic_strength = rng.uniform(0.0, 1.8)
        high_tilt = rng.uniform(-0.4, 0.6)

        spec: List[List[float]] = []
        for f in range(n_freq):
            freq_norm = (f - (n_freq - 1) / 2.0) / n_freq
            is_harmonic_comb = 1.0 if (f % 2 == 0) else 0.0
            row: List[float] = []
            for t in range(n_time):
                val = base_energy + high_tilt * freq_norm
                if t >= onset_t and t <= onset_t + 2:
                    val += onset_burst * (1.0 if t == onset_t else 0.5)
                val += harmonic_strength * is_harmonic_comb * 0.6
                val += float(rng.uniform(0.0, 0.12))
                row.append(max(val, 0.01))
            spec.append(row)
        stimuli.append(np.array(spec, dtype=np.float64))

    X_gt, _ = extract_auditory_ecog_features(stimuli)
    X_gt_list = X_gt.tolist()
    y_rows: List[List[float]] = []
    # Features:
    # 0: broadband_rms_envelope, 1: positive_spectral_flux_onset, 2: spectral_centroid,
    # 3: high_to_low_freq_ratio, 4: harmonic_comb_modulation, 5: temporal_modulation_depth
    for row in X_gt_list:
        rms, onset_flux, centroid, hl_ratio, harm_comb, temp_mod = row
        hg_a1 = 1.9 * rms + 1.4 * centroid + 0.5 * hl_ratio + float(rng.normal(0.0, 0.08))
        stg_onset = 2.4 * onset_flux + 1.1 * temp_mod + 0.2 * rms + float(rng.normal(0.0, 0.08))
        stg_pitch = 2.3 * harm_comb - 0.6 * hl_ratio + 0.4 * rms + float(rng.normal(0.0, 0.08))
        y_rows.append([hg_a1, stg_onset, stg_pitch])

    Y = np.array(y_rows, dtype=np.float64)
    region_names = ["HG_Primary_A1", "STG_Onset_Transient", "STG_Harmonic_Pitch"]
    return (
        stimuli[:n_train],
        Y[:n_train],
        stimuli[n_train:],
        Y[n_train:],
        region_names,
    )


def extract_auditory_ecog_features(stimuli: Sequence[Any]) -> Tuple[Any, List[str]]:
    """
    Hand-programmed interpretable Auditory ECoG feature bank discovered by iWSP:
    1. `broadband_rms_envelope`: Root-mean-square acoustic intensity across time-frequency bins.
    2. `positive_spectral_flux_onset`: Mean half-wave rectified temporal derivative
       $\max(0, S_{f, t} - S_{f, t-1})$ capturing sudden acoustic onsets in STG.
    3. `spectral_centroid`: Energy-weighted center of frequency mass $\frac{\sum_f f \cdot E_f}{\sum_f E_f}$.
    4. `high_to_low_freq_ratio`: Log energy ratio of upper vs lower frequency bands.
    5. `harmonic_comb_modulation`: Even-vs-odd spectral band contrast measuring harmonicity / pitch structure.
    6. `temporal_modulation_depth`: Peak-to-mean temporal envelope ratio across time frames.
    """
    feature_names = [
        "broadband_rms_envelope",
        "positive_spectral_flux_onset",
        "spectral_centroid",
        "high_to_low_freq_ratio",
        "harmonic_comb_modulation",
        "temporal_modulation_depth",
    ]
    rows: List[List[float]] = []

    for stim in stimuli:
        S = stim.tolist() if hasattr(stim, "tolist") else stim
        F = len(S)
        T = len(S[0])
        flat = [S[f][t] for f in range(F) for t in range(T)]
        rms = math.sqrt(sum(x * x for x in flat) / len(flat))

        # Half-wave rectified positive spectral flux (acoustic onset detector)
        pos_flux = [
            max(0.0, S[f][t] - S[f][t - 1])
            for f in range(F)
            for t in range(1, T)
        ]
        onset_flux = max(pos_flux) + (sum(pos_flux) / max(len(pos_flux), 1))

        # Spectral centroid
        band_energies = [sum(S[f][t] for t in range(T)) / T for f in range(F)]
        tot_e = sum(band_energies) + 1e-8
        centroid = sum(float(f) * band_energies[f] for f in range(F)) / (tot_e * F)

        # High-to-low frequency ratio
        mid_f = F // 2
        low_e = sum(band_energies[:mid_f]) + 1e-6
        high_e = sum(band_energies[mid_f:]) + 1e-6
        hl_ratio = math.log(high_e / low_e)

        # Harmonic comb modulation (alternating harmonic peak contrast)
        even_e = sum(band_energies[f] for f in range(0, F, 2)) / max((F + 1) // 2, 1)
        odd_e = sum(band_energies[f] for f in range(1, F, 2)) / max(F // 2, 1)
        harm_comb = (even_e - odd_e) / ( (even_e + odd_e) * 0.5 + 1e-6 )

        # Temporal envelope modulation depth
        frame_env = [sum(S[f][t] for f in range(F)) / F for t in range(T)]
        mean_env = (sum(frame_env) / T) + 1e-6
        temp_mod = (max(frame_env) - min(frame_env)) / mean_env

        rows.append([
            rms,
            onset_flux,
            centroid,
            hl_ratio,
            harm_comb,
            temp_mod,
        ])

    return np.array(rows, dtype=np.float64), feature_names


# ===========================================================================
# Evaluation & Offline iWSP Trajectories for Brain Encoding
# ===========================================================================

BRAIN_MODALITIES: Dict[str, Tuple[Callable[..., Any], Callable[[Any], Tuple[Any, List[str]]]]] = {
    "visual_cortex": (generate_visual_cortex_dataset, extract_visual_cortex_features),
    "language_fmri": (generate_language_fmri_dataset, extract_language_fmri_features),
    "auditory_ecog": (generate_auditory_ecog_dataset, extract_auditory_ecog_features),
}


def evaluate_brain_modality(
    modality: str,
    feature_fn: Optional[Callable[[Any], Tuple[Any, List[str]]]] = None,
) -> Tuple[float, Dict[str, Any], str, RidgeEncodingReadout]:
    """Evaluate a programmed feature space on one of the 3 brain encoding modalities."""
    if modality not in BRAIN_MODALITIES:
        raise ValueError(f"Unknown brain modality '{modality}'. Choose from {list(BRAIN_MODALITIES.keys())}")
    gen_fn, default_feat_fn = BRAIN_MODALITIES[modality]
    stim_tr, Y_tr, stim_te, Y_te, regions = gen_fn()
    evaluator = BrainEncodingEvaluator(stim_tr, Y_tr, stim_te, Y_te, regions)
    return evaluator.evaluate(feature_fn or default_feat_fn)


def get_brain_offline_trajectories() -> Dict[str, List[Tuple[str, str]]]:
    """
    Provide multi-step iWSP scientific refinement trajectories for the 3 neuroscience domains:
    - `visual_cortex`: Progresses from low-level global luminance/contrast to V1 oriented edges +
      V4/IT warm-cool color opponency and foveal center-surround saliency.
    - `language_fmri`: Progresses from basic lexical word count to semantic/syntactic categories,
      and finally discovers the scan-edge timing confound (`scan_edge_confound`) that unlocks
      high prediction accuracy in `Scan_Boundary_ROI`.
    - `auditory_ecog`: Progresses from static broadband RMS energy to half-wave rectified spectral
      flux onsets and harmonic comb periodicity.
    """
    return {
        "visual_cortex": [
            (
                "Attempt 1: Global luminance and RMS contrast only (misses V1 orientation selectivity and V4 color/figure-ground tuning).",
                """
def extract_features(stimuli):
    rows = []
    for stim in stimuli:
        img = stim.tolist() if hasattr(stim, "tolist") else stim
        H, W = len(img), len(img[0])
        flat = [(img[r][c][0] + img[r][c][1] + img[r][c][2]) / 3.0 for r in range(H) for c in range(W)]
        m = sum(flat) / len(flat)
        c = math.sqrt(sum((x - m) ** 2 for x in flat) / len(flat))
        rows.append([m, c])
    return np.array(rows), ["mean_luminance", "rms_contrast"]
""",
            ),
            (
                "Attempt 2: Add V1 horizontal and vertical spatial derivative edge energy (drastically improves V1_Early r, but V4_IT_Ventral remains under-explained).",
                """
def extract_features(stimuli):
    from iwsp.brain.modalities import extract_visual_cortex_features
    X, names = extract_visual_cortex_features(stimuli)
    return X[:, :4], names[:4]
""",
            ),
            (
                "Attempt 3: Add V4/IT warm-cool chromatic opponency (R-B), foveal center-surround saliency, and bilateral mirror symmetry to capture ventral stream computations.",
                """
def extract_features(stimuli):
    from iwsp.brain.modalities import extract_visual_cortex_features
    return extract_visual_cortex_features(stimuli)
""",
            ),
        ],
        "language_fmri": [
            (
                "Attempt 1: Surface lexical features only (word_rate and mean_word_length); fails on semantic, syntactic, and scan-boundary ROIs.",
                """
def extract_features(stimuli):
    from iwsp.brain.modalities import extract_language_fmri_features
    X, names = extract_language_fmri_features(stimuli)
    return X[:, :2], names[:2]
""",
            ),
            (
                "Attempt 2: Add social/animate semantic density, spatial scene density, and syntactic subordination density (solves Temp_Semantic and IFG_Syntactic, but Scan_Boundary_ROI remains near r=0).",
                """
def extract_features(stimuli):
    from iwsp.brain.modalities import extract_language_fmri_features
    X, names = extract_language_fmri_features(stimuli)
    return X[:, :5], names[:5]
""",
            ),
            (
                "Attempt 3: Diagnose fMRI run boundary residuals and program explicit scan-edge timing confound (exponential decay at start/end TRs of run), surfacing the scanner timing artifact in Scan_Boundary_ROI.",
                """
def extract_features(stimuli):
    from iwsp.brain.modalities import extract_language_fmri_features
    return extract_language_fmri_features(stimuli)
""",
            ),
        ],
        "auditory_ecog": [
            (
                "Attempt 1: Static broadband RMS energy and spectral centroid (captures HG_Primary_A1 tonotopy/loudness, but misses STG onset transients and harmonic pitch).",
                """
def extract_features(stimuli):
    from iwsp.brain.modalities import extract_auditory_ecog_features
    X, names = extract_auditory_ecog_features(stimuli)
    return np.column_stack([X[:, 0], X[:, 2]]), [names[0], names[2]]
""",
            ),
            (
                "Attempt 2: Add half-wave rectified positive spectral flux max(0, S[t] - S[t-1]) and harmonic comb periodicity to predict STG_Onset_Transient and STG_Harmonic_Pitch.",
                """
def extract_features(stimuli):
    from iwsp.brain.modalities import extract_auditory_ecog_features
    return extract_auditory_ecog_features(stimuli)
""",
            ),
        ],
    }
