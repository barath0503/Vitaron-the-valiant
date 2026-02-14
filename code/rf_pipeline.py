"""
RF Dielectric Sensing Pipeline
=========================================
Stages:
1. Load S11 sweep from tb CSV
2. Resonance detection
3. Phase unwrap + smoothing
4. Multi-frame accumulation
5. Feature extraction for fusion
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.signal import find_peaks


# =============================================================================
# PATH RESOLUTION (NEW)
# =============================================================================

def resolve_csv_path(filename: str) -> str:
    """
    Structure:
    VITARON-THE-VALIANT/
        code/
            rf_pipeline.py
        tb/
            synthetic_rf_sweep.csv
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))   # code/
    project_root = os.path.dirname(current_dir)                # repo root
    tb_folder = os.path.join(project_root, "tb")

    return os.path.join(tb_folder, filename)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RFSweep:
    frequency_mhz: np.ndarray
    s11_magnitude_db: np.ndarray
    s11_phase_deg: np.ndarray
    metadata: Dict | None = None


@dataclass
class RFFeatures:
    effective_permittivity: float
    phase_delay: float
    hydration_index: float
    skin_thickness_estimate: float
    debug: Dict | None = None


# =============================================================================
# LOADING
# =============================================================================

def load_rf_csv(filename: str) -> RFSweep:
    csv_path = resolve_csv_path(filename)
    df = pd.read_csv(csv_path)

    return RFSweep(
        frequency_mhz=df["frequency_mhz"].to_numpy(),
        s11_magnitude_db=df["s11_magnitude_db"].to_numpy(),
        s11_phase_deg=df["s11_phase_deg"].to_numpy(),
        metadata={"source": "csv", "path": csv_path},
    )


# =============================================================================
# PROCESSING UTILITIES
# =============================================================================

def find_resonance_peak(sweep: RFSweep) -> Tuple[float, float]:

    peaks, _ = find_peaks(-sweep.s11_magnitude_db, height=-25, distance=10)

    if len(peaks) == 0:
        return 300.0, 10.0

    f_res = sweep.frequency_mhz[peaks[0]]

    half_power_idx = np.where(sweep.s11_magnitude_db > -3)[0]

    if len(half_power_idx) > 1:
        bandwidth = sweep.frequency_mhz[half_power_idx[-1]] - sweep.frequency_mhz[half_power_idx[0]]
        q_factor = f_res / bandwidth if bandwidth > 0 else 10.0
    else:
        q_factor = 10.0

    return float(f_res), float(q_factor)


def preprocess_rf_sweep(sweep: RFSweep):

    # Smooth magnitude (moving average)
    window = min(21, max(5, len(sweep.s11_magnitude_db) // 10))
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    s11_smooth = np.convolve(sweep.s11_magnitude_db, kernel, mode="same")

    # Phase unwrap
    phase_unwrapped = np.unwrap(sweep.s11_phase_deg * np.pi / 180) * 180 / np.pi

    cleaned = RFSweep(
        sweep.frequency_mhz.copy(),
        s11_smooth,
        phase_unwrapped,
        sweep.metadata,
    )

    return cleaned, {"raw_resonance": find_resonance_peak(sweep)}


# =============================================================================
# FRAME ACCUMULATION
# =============================================================================

def accumulate_rf_frames(frames: List[RFSweep]):

    mags = np.stack([f.s11_magnitude_db for f in frames], axis=0)
    phases = np.stack([f.s11_phase_deg for f in frames], axis=0)

    mean_mag = np.mean(mags, axis=0)
    mean_phase = np.mean(phases, axis=0)

    frame_var = float(np.mean(np.var(mags, axis=0)))

    return (
        RFSweep(frames[0].frequency_mhz.copy(), mean_mag, mean_phase),
        frame_var,
    )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_rf_features(sweep: RFSweep, debug_info=None):

    f_res, q_factor = find_resonance_peak(sweep)

    # Simple effective permittivity model
    effective_permittivity = float(2500 / (f_res ** 2) * 1e6 + 30)

    # Phase delay at resonance
    res_idx = np.argmin(np.abs(sweep.frequency_mhz - f_res))
    phase_delay = float(abs(sweep.s11_phase_deg[res_idx]))

    hydration_index = float(np.clip((effective_permittivity - 40) / 40, 0.0, 1.0))
    skin_thickness = float(phase_delay / 12.5)

    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info else 0.0

    return RFFeatures(
        effective_permittivity,
        phase_delay,
        hydration_index,
        skin_thickness,
        debug={
            "resonance_mhz": f_res,
            "q_factor": q_factor,
            "frame_variance": frame_var,
        },
    )


# =============================================================================
# MAIN PIPELINE ENTRY
# =============================================================================

def rf_pipeline_from_csv(filename: str):

    base_sweep = load_rf_csv(filename)

    raw_frames = [
        RFSweep(
            base_sweep.frequency_mhz,
            base_sweep.s11_magnitude_db + np.random.normal(0, 0.5, base_sweep.s11_magnitude_db.shape),
            base_sweep.s11_phase_deg + np.random.normal(0, 2, base_sweep.s11_phase_deg.shape),
        )
        for _ in range(8)
    ]

    processed = []
    for f in raw_frames:
        cleaned, _ = preprocess_rf_sweep(f)
        processed.append(cleaned)

    acc_sweep, frame_var = accumulate_rf_frames(processed)

    acc_cleaned, dbg = preprocess_rf_sweep(acc_sweep)
    dbg["frame_variance"] = frame_var

    return extract_rf_features(acc_cleaned, dbg)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":

    rf_feats = rf_pipeline_from_csv("synthetic_rf_sweep.csv")

    print("RF Dielectric Sensing Feature Vector:")
    print(f"Permittivity: {rf_feats.effective_permittivity:.1f}")
    print(f"Phase delay: {rf_feats.phase_delay:.1f}°")
    print(f"Hydration index: {rf_feats.hydration_index:.3f}")
    print(f"Skin thickness: {rf_feats.skin_thickness_estimate:.1f} mm")
