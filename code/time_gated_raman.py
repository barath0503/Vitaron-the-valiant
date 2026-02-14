"""
TIME-GATED RAMAN PIPELINE
Stages:
1. Load raw spectrum (from tb CSV)
2. Time-gating (fluorescence suppression)
3. Baseline correction (ALS)
4. Denoising (Savitzky–Golay)
5. Frame accumulation
6. Feature extraction
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix


# =============================================================================
# PATH RESOLUTION (NEW)
# =============================================================================

def resolve_csv_path(filename: str) -> str:
    """
    Structure:
    VITARON-THE-VALIANT/
        code/
            time_gated_raman.py
        tb/
            synthetic_glucose_raman_spectrum.csv
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))   # code/
    project_root = os.path.dirname(current_dir)                # repo root
    tb_folder = os.path.join(project_root, "tb")

    return os.path.join(tb_folder, filename)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RamanSpectrum:
    shift: np.ndarray
    intensity: np.ndarray
    metadata: Dict | None = None


@dataclass
class RamanFeatures:
    peak_intensity_1125: float
    peak_intensity_1340: float
    peak_intensity_1460: float

    area_1125: float
    area_1340: float
    area_1460: float

    ratio_1125_to_1340: float
    ratio_1340_to_1460: float

    snr_estimate: float
    fluorescence_index: float
    frame_variance: float

    debug: Dict | None = None


# =============================================================================
# LOADING
# =============================================================================

def load_raman_csv(filename: str) -> RamanSpectrum:
    csv_path = resolve_csv_path(filename)

    df = pd.read_csv(csv_path)

    shift = df["raman_shift_cm^-1"].to_numpy()
    intensity = df["intensity_raw"].to_numpy()

    return RamanSpectrum(
        shift=shift,
        intensity=intensity,
        metadata={"source": "csv", "path": csv_path},
    )


# =============================================================================
# BASELINE CORRECTION (ALS)
# =============================================================================

def asymmetric_least_squares_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.001,
    niter: int = 10,
) -> np.ndarray:

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)

    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = csc_matrix(W + D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# =============================================================================
# SIGNAL UTILITIES
# =============================================================================

def apply_time_gating(intensity: np.ndarray, gate_width: float = 0.3):
    threshold = np.quantile(intensity, 1 - gate_width)
    return np.minimum(intensity, threshold)


def compute_snr(signal: np.ndarray, region_mask: np.ndarray) -> float:
    if not np.any(region_mask):
        return 0.0

    signal_level = np.mean(signal[region_mask])
    noise_std = np.std(signal[~region_mask]) if np.any(~region_mask) else 1e-9

    return float(signal_level / noise_std)


def integrate_region(shift, intensity, mask):
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(intensity[mask], shift[mask]))


def peak_max_in_window(shift, intensity, center, half_width=20.0):
    mask = (shift >= center - half_width) & (shift <= center + half_width)
    if not np.any(mask):
        return 0.0
    return float(np.max(intensity[mask]))


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_single_spectrum(spec: RamanSpectrum):

    y = spec.intensity.copy()

    # 1. Time gating
    y_gated = apply_time_gating(y)

    # 2. Baseline correction
    baseline = asymmetric_least_squares_baseline(y_gated)
    y_corrected = y_gated - baseline

    # 3. Denoising
    sg_window = min(15, len(y_corrected) - 1)
    if sg_window % 2 == 0:
        sg_window -= 1
    if sg_window < 5:
        sg_window = 5

    y_denoised = savgol_filter(y_corrected, sg_window, 3)

    cleaned = RamanSpectrum(
        shift=spec.shift.copy(),
        intensity=y_denoised,
        metadata=spec.metadata,
    )

    debug = {
        "baseline": baseline,
        "frame_variance": 0.0,
    }

    return cleaned, debug


# =============================================================================
# ACCUMULATION
# =============================================================================

def accumulate_frames(frames: List[RamanSpectrum]):

    intensities = np.stack([f.intensity for f in frames], axis=0)
    mean_intensity = np.mean(intensities, axis=0)
    frame_var = float(np.mean(np.var(intensities, axis=0)))

    return (
        RamanSpectrum(frames[0].shift.copy(), mean_intensity),
        frame_var,
    )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_glucose_features(spec: RamanSpectrum, debug_info=None):

    shift = spec.shift
    intensity = spec.intensity

    def mask(center):
        return (shift >= center - 20) & (shift <= center + 20)

    peak_1125 = peak_max_in_window(shift, intensity, 1125)
    peak_1340 = peak_max_in_window(shift, intensity, 1340)
    peak_1460 = peak_max_in_window(shift, intensity, 1460)

    area_1125 = integrate_region(shift, intensity, mask(1125))
    area_1340 = integrate_region(shift, intensity, mask(1340))
    area_1460 = integrate_region(shift, intensity, mask(1460))

    eps = 1e-9
    ratio_1125_to_1340 = peak_1125 / (peak_1340 + eps)
    ratio_1340_to_1460 = peak_1340 / (peak_1460 + eps)

    signal_mask = mask(1125) | mask(1340) | mask(1460)
    snr = compute_snr(intensity, signal_mask)

    fluorescence_index = float(
        np.sqrt(np.mean(debug_info["baseline"] ** 2))
    ) if debug_info else 0.0

    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info else 0.0

    return RamanFeatures(
        peak_1125,
        peak_1340,
        peak_1460,
        area_1125,
        area_1340,
        area_1460,
        ratio_1125_to_1340,
        ratio_1340_to_1460,
        snr,
        fluorescence_index,
        frame_var,
        debug_info,
    )


# =============================================================================
# MAIN PIPELINE ENTRY
# =============================================================================

def raman_pipeline_from_csv(filename: str):

    base_spec = load_raman_csv(filename)

    raw_frames = [
        RamanSpectrum(
            base_spec.shift,
            base_spec.intensity + np.random.normal(0, 0.01, size=base_spec.intensity.shape),
        )
        for _ in range(8)
    ]

    processed = []
    for f in raw_frames:
        cleaned, _ = preprocess_single_spectrum(f)
        processed.append(cleaned)

    acc_spec, frame_var = accumulate_frames(processed)

    acc_cleaned, dbg = preprocess_single_spectrum(acc_spec)
    dbg["frame_variance"] = frame_var

    return extract_glucose_features(acc_cleaned, dbg)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":

    rfeats = raman_pipeline_from_csv("synthetic_glucose_raman_spectrum.csv")

    print("Time-gated RAMAN Feature Vector:")
    print(f"Peak 1125: {rfeats.peak_intensity_1125:.3f}")
    print(f"Peak 1340: {rfeats.peak_intensity_1340:.3f}")
    print(f"Peak 1460: {rfeats.peak_intensity_1460:.3f}")
    print(f"SNR: {rfeats.snr_estimate:.3f}")
