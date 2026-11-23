"""
2D Radio Wave Propagation & Confound Discovery Encoding Model via iWSP
======================================================================

Applies Interpretable Weight-Space Programming (iWSP) to physics-based wireless
radio propagation in a 100m x 100m indoor / urban-microcell environment.

Reproduces the iWSP brain-encoding paradigm in a physical RF domain:
- Instead of training a black-box neural network on raw coordinates, the iWSP
  agent programs explicit geometric and electromagnetic features (Friis path loss,
  2D ray-casting wall penetration, Two-Ray Ground Reflection breakpoint & phase
  interference, sector antenna pattern, and spatial anomaly/interference confounds),
  paired with a cross-validated Ridge Regression readout (`RidgeEncodingReadout`).

Three evaluated radio channels (dBm):
1. `Sub6_Indoor_RSSI`: Dominated by Friis log-distance/frequency path loss and
   2D line-of-sight concrete / drywall penetration attenuation.
2. `mmWave_28GHz_RSRP`: Strongly affected by Two-Ray Ground Reflection breakpoint
   excess loss `ReLU(log10(d / d_bp))`, phase interference, and heavy concrete blockage.
3. `Industrial_Sector_RSSI`: Combines directional sector antenna boresight gain
   with an initially unknown localized interference/attenuation anomaly near `(75, 25)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from iwsp.brain.ridge import BrainEncodingEvaluator, RidgeEncodingReadout
from iwsp.math_backend import np

SPEED_OF_LIGHT_MPS = 3.0e8
DEFAULT_TX_POS: Tuple[float, float] = (15.0, 20.0)
DEFAULT_BORESIGHT_RAD: float = math.pi / 4.0  # 45 deg northeast into room
DEFAULT_ANOMALY_POS: Tuple[float, float] = (75.0, 25.0)
DEFAULT_ANOMALY_SIGMA_M: float = 16.0


@dataclass(frozen=True)
class WallSegment:
    """2D wall obstacle segment with material classification."""

    x1: float
    y1: float
    x2: float
    y2: float
    material: str  # "concrete" or "drywall"


DEFAULT_WALLS: Tuple[WallSegment, ...] = (
    WallSegment(35.0, 10.0, 35.0, 75.0, "concrete"),
    WallSegment(35.0, 65.0, 90.0, 65.0, "concrete"),
    WallSegment(65.0, 10.0, 65.0, 65.0, "drywall"),
    WallSegment(10.0, 38.0, 65.0, 38.0, "drywall"),
    WallSegment(20.0, 55.0, 20.0, 95.0, "drywall"),
)


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    """2D cross product (b - a) x (c - a)."""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> bool:
    return (
        min(ax, bx) - 1e-9 <= px <= max(ax, bx) + 1e-9
        and min(ay, by) - 1e-9 <= py <= max(ay, by) + 1e-9
    )


def segments_intersect(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    """Return True if closed 2D line segments AB and CD intersect."""
    o1 = _orientation(ax, ay, bx, by, cx, cy)
    o2 = _orientation(ax, ay, bx, by, dx, dy)
    o3 = _orientation(cx, cy, dx, dy, ax, ay)
    o4 = _orientation(cx, cy, dx, dy, bx, by)

    if (o1 > 1e-9 and o2 < -1e-9 or o1 < -1e-9 and o2 > 1e-9) and (
        o3 > 1e-9 and o4 < -1e-9 or o3 < -1e-9 and o4 > 1e-9
    ):
        return True
    if abs(o1) <= 1e-9 and _on_segment(ax, ay, bx, by, cx, cy):
        return True
    if abs(o2) <= 1e-9 and _on_segment(ax, ay, bx, by, dx, dy):
        return True
    if abs(o3) <= 1e-9 and _on_segment(cx, cy, dx, dy, ax, ay):
        return True
    if abs(o4) <= 1e-9 and _on_segment(cx, cy, dx, dy, bx, by):
        return True
    return False


def count_wall_crossings(
    tx_pos: Tuple[float, float],
    rx_pos: Tuple[float, float],
    walls: Sequence[WallSegment],
) -> Tuple[int, int]:
    """Count `(concrete_crossings, drywall_crossings)` along the Tx-Rx line-of-sight ray."""
    n_concrete = 0
    n_drywall = 0
    for w in walls:
        if segments_intersect(
            tx_pos[0], tx_pos[1], rx_pos[0], rx_pos[1], w.x1, w.y1, w.x2, w.y2
        ):
            if w.material == "concrete":
                n_concrete += 1
            else:
                n_drywall += 1
    return n_concrete, n_drywall


def extract_radio_propagation_features(
    stimuli: Sequence[Dict[str, Any]],
) -> Tuple[Any, List[str]]:
    """
    Hand-programmed interpretable RF physical feature bank discovered by iWSP:
    1. `log10_distance_3d`: log10 of 3D Tx-Rx separation distance (Friis distance decay).
    2. `log10_freq_ghz`: log10 of carrier frequency in GHz (aperture frequency loss).
    3. `concrete_wall_crossings`: Number of concrete walls intersected by LoS ray.
    4. `drywall_crossings`: Number of drywall partitions intersected by LoS ray.
    5. `post_breakpoint_excess_log_d`: ReLU(log10(d_2d / d_bp)) beyond Two-Ray breakpoint.
    6. `two_ray_phase_interference`: cos(4 * pi * h_t * h_r / (lambda * d_2d)).
    7. `tx_sector_boresight_gain`: cos(azimuth(rx - tx) - boresight_angle).
    8. `industrial_anomaly_zone_confound`: Localized spatial Gaussian attenuation near (75, 25).
    """
    feature_names = [
        "log10_distance_3d",
        "log10_freq_ghz",
        "concrete_wall_crossings",
        "drywall_crossings",
        "post_breakpoint_excess_log_d",
        "two_ray_phase_interference",
        "tx_sector_boresight_gain",
        "industrial_anomaly_zone_confound",
    ]
    rows: List[List[float]] = []
    for item in stimuli:
        tx_x, tx_y = item["tx_pos"]
        rx_x, rx_y = item["rx_pos"]
        ht = float(item["ht_m"])
        hr = float(item["hr_m"])
        freq_ghz = float(item["freq_ghz"])
        boresight = float(item.get("tx_boresight_rad", DEFAULT_BORESIGHT_RAD))
        walls = item.get("walls", DEFAULT_WALLS)

        dx = float(rx_x - tx_x)
        dy = float(rx_y - tx_y)
        dz = ht - hr
        d_2d = max(math.hypot(dx, dy), 1.0)
        d_3d = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0)

        log_d3d = math.log10(d_3d)
        log_f = math.log10(max(freq_ghz, 0.1))

        n_conc, n_dry = count_wall_crossings((tx_x, tx_y), (rx_x, rx_y), walls)

        wavelength = SPEED_OF_LIGHT_MPS / (freq_ghz * 1.0e9)
        d_bp = max((4.0 * ht * hr) / wavelength, 1.0)
        post_bp = max(0.0, math.log10(d_2d) - math.log10(d_bp))
        two_ray_phase = math.cos((4.0 * math.pi * ht * hr) / (wavelength * d_2d))

        az = math.atan2(dy, dx)
        sector_gain = math.cos(az - boresight)

        ax, ay = DEFAULT_ANOMALY_POS
        dist_anom_sq = (rx_x - ax) ** 2 + (rx_y - ay) ** 2
        anomaly_confound = math.exp(-dist_anom_sq / (2.0 * DEFAULT_ANOMALY_SIGMA_M ** 2))

        rows.append([
            log_d3d,
            log_f,
            float(n_conc),
            float(n_dry),
            post_bp,
            two_ray_phase,
            sector_gain,
            anomaly_confound,
        ])

    return np.array(rows, dtype=np.float64), feature_names


def generate_radio_propagation_dataset(
    n_train: int = 120, n_test: int = 60, seed: int = 2026
) -> Tuple[List[Dict[str, Any]], Any, List[Dict[str, Any]], Any, List[str]]:
    """
    Generate 2D radio propagation measurement samples across a 100m x 100m floorplan
    with 3 target received-power channels (`Sub6_Indoor_RSSI`, `mmWave_28GHz_RSRP`,
    `Industrial_Sector_RSSI`).
    """
    rng = np.random.RandomState(seed)
    total = n_train + n_test
    freq_choices = [0.9, 1.8, 2.4, 3.5, 5.8, 28.0]
    stimuli: List[Dict[str, Any]] = []

    while len(stimuli) < total:
        rx_x = float(rng.uniform(4.0, 96.0))
        rx_y = float(rng.uniform(4.0, 96.0))
        if math.hypot(rx_x - DEFAULT_TX_POS[0], rx_y - DEFAULT_TX_POS[1]) < 3.0:
            continue
        freq_ghz = freq_choices[int(rng.randint(0, len(freq_choices)))]
        ht_m = float(rng.uniform(1.5, 2.6))
        hr_m = float(rng.uniform(0.9, 1.5))
        stimuli.append({
            "tx_pos": DEFAULT_TX_POS,
            "rx_pos": (rx_x, rx_y),
            "ht_m": ht_m,
            "hr_m": hr_m,
            "freq_ghz": freq_ghz,
            "tx_boresight_rad": DEFAULT_BORESIGHT_RAD,
            "walls": DEFAULT_WALLS,
        })

    X_gt, _ = extract_radio_propagation_features(stimuli)
    X_list = X_gt.tolist()
    y_rows: List[List[float]] = []

    for row in X_list:
        log_d, log_f, n_conc, n_dry, post_bp, two_ray, sector_g, anom = row
        sub6_rssi = (
            -32.4
            - 22.0 * log_d
            - 16.0 * log_f
            - 12.0 * n_conc
            - 4.5 * n_dry
            + float(rng.normal(0.0, 0.3))
        )
        mmwave_rsrp = (
            -40.0
            - 20.0 * log_d
            - 18.0 * log_f
            - 15.0 * n_conc
            - 30.0 * post_bp
            + 5.5 * two_ray
            + float(rng.normal(0.0, 0.3))
        )
        industrial_rssi = (
            -35.0
            - 20.0 * log_d
            - 10.0 * log_f
            - 8.0 * n_conc
            + 18.0 * sector_g
            - 24.0 * anom
            + float(rng.normal(0.0, 0.3))
        )
        y_rows.append([sub6_rssi, mmwave_rsrp, industrial_rssi])

    Y = np.array(y_rows, dtype=np.float64)
    channel_names = [
        "Sub6_Indoor_RSSI",
        "mmWave_28GHz_RSRP",
        "Industrial_Sector_RSSI",
    ]
    return (
        stimuli[:n_train],
        Y[:n_train],
        stimuli[n_train:],
        Y[n_train:],
        channel_names,
    )


def evaluate_radio_propagation(
    feature_fn: Optional[Callable[[Any], Tuple[Any, List[str]]]] = None,
) -> Tuple[float, Dict[str, Any], str, RidgeEncodingReadout]:
    """Evaluate a programmed RF feature extractor via cross-validated Ridge Regression."""
    stim_tr, Y_tr, stim_te, Y_te, channels = generate_radio_propagation_dataset()
    evaluator = BrainEncodingEvaluator(stim_tr, Y_tr, stim_te, Y_te, channels)
    return evaluator.evaluate(feature_fn or extract_radio_propagation_features)


def render_ascii_radio_heatmap(
    readout: Optional[RidgeEncodingReadout] = None,
    channel_idx: int = 2,
    freq_ghz: float = 2.4,
    width: int = 44,
    height: int = 20,
) -> str:
    """
    Render a 2D ASCII signal-strength heatmap across the 100m x 100m floorplan
    showing the transmitter (`T`), concrete/drywall obstacles (`#`), and predicted
    received power levels (` .:-=+*#%@` from weak to strong).
    """
    if readout is None:
        _, _, _, readout = evaluate_radio_propagation()

    grid_stimuli: List[Dict[str, Any]] = []
    coords: List[Tuple[int, int, float, float]] = []
    for row_i in range(height):
        # Top row corresponds to y = 97.5m, bottom row to y = 2.5m
        y_m = 100.0 * (1.0 - (row_i + 0.5) / height)
        for col_i in range(width):
            x_m = 100.0 * ((col_i + 0.5) / width)
            coords.append((row_i, col_i, x_m, y_m))
            grid_stimuli.append({
                "tx_pos": DEFAULT_TX_POS,
                "rx_pos": (x_m, y_m),
                "ht_m": 2.0,
                "hr_m": 1.2,
                "freq_ghz": freq_ghz,
                "tx_boresight_rad": DEFAULT_BORESIGHT_RAD,
                "walls": DEFAULT_WALLS,
            })

    X_grid, _ = extract_radio_propagation_features(grid_stimuli)
    preds = readout.predict(X_grid).tolist()
    vals = [float(p[channel_idx]) for p in preds]
    v_min = min(vals)
    v_max = max(vals)
    span = max(v_max - v_min, 1e-6)
    palette = " .:-=+*%@"

    canvas: List[List[str]] = [[" " for _ in range(width)] for _ in range(height)]
    for (r_i, c_i, x_m, y_m), val in zip(coords, vals):
        norm = (val - v_min) / span
        p_idx = min(int(norm * (len(palette) - 1)), len(palette) - 1)
        canvas[r_i][c_i] = palette[p_idx]

    # Overlay walls (#)
    cell_w = 100.0 / width
    cell_h = 100.0 / height
    for w in DEFAULT_WALLS:
        steps = 60
        for s in range(steps + 1):
            t = s / steps
            wx = w.x1 + t * (w.x2 - w.x1)
            wy = w.y1 + t * (w.y2 - w.y1)
            c_i = min(max(int(wx / cell_w), 0), width - 1)
            r_i = min(max(int((100.0 - wy) / cell_h), 0), height - 1)
            canvas[r_i][c_i] = "#"

    # Overlay Transmitter (T) and Anomaly Center (X)
    tx_c = min(max(int(DEFAULT_TX_POS[0] / cell_w), 0), width - 1)
    tx_r = min(max(int((100.0 - DEFAULT_TX_POS[1]) / cell_h), 0), height - 1)
    canvas[tx_r][tx_c] = "T"

    ax_c = min(max(int(DEFAULT_ANOMALY_POS[0] / cell_w), 0), width - 1)
    ax_r = min(max(int((100.0 - DEFAULT_ANOMALY_POS[1]) / cell_h), 0), height - 1)
    canvas[ax_r][ax_c] = "X"

    ch_name = readout.region_names[channel_idx] if channel_idx < len(readout.region_names) else f"Channel {channel_idx}"
    lines = [
        f"2D Floorplan Predicted Heatmap ({ch_name} @ {freq_ghz:.1f} GHz)",
        f"Range: [{v_min:.1f} dBm (weak '.') .. {v_max:.1f} dBm (strong '@')] | T=Tx, #=Wall, X=Anomaly(75,25)",
        "+" + "-" * width + "+",
    ]
    for row_chars in canvas:
        lines.append("|" + "".join(row_chars) + "|")
    lines.append("+" + "-" * width + "+")
    return "\n".join(lines)


def get_radio_propagation_offline_trajectory() -> Dict[str, List[Tuple[str, str]]]:
    """
    Provide a 3-step iWSP scientific discovery trajectory for 2D radio propagation:
    - Iteration 1: Free-space Friis path loss only (log10 distance & log10 frequency).
    - Iteration 2: Add 2D ray-casting wall penetrations + Two-Ray Ground Reflection
      breakpoint excess loss and phase interference.
    - Iteration 3: Discover directional sector antenna boresight gain and localized
      industrial interference anomaly at (75, 25) from validation residuals.
    """
    return {
        "radio_propagation": [
            (
                "Attempt 1: Baseline Friis free-space path loss using only log10(distance_3d) and log10(freq_ghz); ignores indoor walls, ground reflections, and sector/interference patterns.",
                """
def extract_features(stimuli):
    from iwsp.radio.propagation import extract_radio_propagation_features
    X, names = extract_radio_propagation_features(stimuli)
    return X[:, :2], names[:2]
""",
            ),
            (
                "Attempt 2: Add 2D geometric ray-casting for concrete/drywall crossings plus Two-Ray Ground Reflection breakpoint ReLU(log10(d/d_bp)) and phase interference cos(4*pi*ht*hr/(lambda*d)).",
                """
def extract_features(stimuli):
    from iwsp.radio.propagation import extract_radio_propagation_features
    X, names = extract_radio_propagation_features(stimuli)
    return X[:, :6], names[:6]
""",
            ),
            (
                "Attempt 3: Inspect Industrial_Sector_RSSI residual patterns and program Tx sector boresight cosine gain plus localized spatial Gaussian confound at (75, 25).",
                """
def extract_features(stimuli):
    from iwsp.radio.propagation import extract_radio_propagation_features
    return extract_radio_propagation_features(stimuli)
""",
            ),
        ]
    }
