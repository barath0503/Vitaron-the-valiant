"""
Multi-Wavelength NIR Spectroscopy Pipeline 
=====================================================
VITARON NIR module - LOADS FROM EXISTING CSV .
Processes 4-wavelength reflectance → 7 features for fusion.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.stats import linregress


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NIRSignal:
    """Raw 4-wavelength photodiode voltages"""
    voltages: np.ndarray
    wavelengths_nm: np.ndarray
    metadata: Dict | None = None


@dataclass
class NIRFeatures:
    """7 features for fusion layer"""
    absorbance_750: float
    absorbance_810: float
    absorbance_940: float
    absorbance_1600: float
    ratio_1600_to_940: float
    absorption_slope: float
    water_band_index: float
    debug: Dict | None = None


# =============================================================================
# PATH RESOLUTION (NEW)
# =============================================================================

def resolve_csv_path(filename: str) -> str:
    """
    Structure:
    VITARON-THE-VALIANT/
        code/
            nir_pipeline.py
        tb/
            synthetic_nir_voltages.csv
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))   # code/
    project_root = os.path.dirname(current_dir)                # VITARON-THE-VALIANT/
    tb_folder = os.path.join(project_root, "tb")

    return os.path.join(tb_folder, filename)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_nir_csv(filename: str) -> NIRSignal:
    csv_path = resolve_csv_path(filename)
    df = pd.read_csv(csv_path)

    voltages = df[
        ["voltage_750", "voltage_810", "voltage_940", "voltage_1600"]
    ].to_numpy()[0]

    return NIRSignal(
        voltages=voltages,
        wavelengths_nm=np.array([750, 810, 940, 1600]),
        metadata={"source": "csv", "path": csv_path}
    )


def dark_subtract_nir(signal: NIRSignal, dark_current: np.ndarray) -> NIRSignal:
    clean_voltages = np.maximum(signal.voltages - dark_current, 0.01)
    return NIRSignal(clean_voltages, signal.wavelengths_nm, signal.metadata)


def convert_to_absorbance(signal: NIRSignal, reference_signal: NIRSignal) -> np.ndarray:
    ratio = signal.voltages / reference_signal.voltages
    return -np.log(np.clip(ratio, 0.01, 2.0))


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def preprocess_single_nir_frame(
    signal: NIRSignal,
    dark_current: np.ndarray,
    water_reference: np.ndarray
) -> Tuple[NIRSignal, Dict]:

    clean_signal = dark_subtract_nir(signal, dark_current)

    absorbance = convert_to_absorbance(
        clean_signal,
        NIRSignal(water_reference, signal.wavelengths_nm)
    )

    processed_signal = NIRSignal(absorbance, signal.wavelengths_nm, signal.metadata)

    debug = {
        "raw_voltages": signal.voltages.copy(),
        "clean_voltages": clean_signal.voltages.copy(),
        "absorbance_raw": absorbance.copy()
    }

    return processed_signal, debug


def accumulate_nir_frames(frames: List[NIRSignal]) -> Tuple[NIRSignal, float]:
    if not frames:
        raise ValueError("No frames provided for accumulation.")

    voltages = np.stack([f.voltages for f in frames], axis=0)
    mean_voltages = np.mean(voltages, axis=0)
    frame_var = float(np.mean(np.var(voltages, axis=0)))

    acc_signal = NIRSignal(
        voltages=mean_voltages,
        wavelengths_nm=frames[0].wavelengths_nm.copy(),
        metadata={"n_frames": len(frames)},
    )

    return acc_signal, frame_var


def extract_nir_features(signal: NIRSignal, debug_info: Dict | None = None) -> NIRFeatures:
    abs_vals = signal.voltages

    abs_750 = float(abs_vals[0])
    abs_810 = float(abs_vals[1])
    abs_940 = float(abs_vals[2])
    abs_1600 = float(abs_vals[3])

    ratio_1600_to_940 = float(abs_1600 / (abs_940 + 1e-6))

    slope, _, _, _, _ = linregress(signal.wavelengths_nm, abs_vals)

    water_band_index = float(abs_940 - 0.5 * (abs_750 + abs_810))

    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info else 0.0

    return NIRFeatures(
        absorbance_750=abs_750,
        absorbance_810=abs_810,
        absorbance_940=abs_940,
        absorbance_1600=abs_1600,
        ratio_1600_to_940=ratio_1600_to_940,
        absorption_slope=float(slope),
        water_band_index=water_band_index,
        debug={"frame_variance": frame_var}
    )


# =============================================================================
# SIMULATION & MAIN PIPELINE
# =============================================================================

def simulate_multi_frame_from_single(
    spec: NIRSignal,
    n_frames: int = 10,
    noise_scale: float = 0.015
) -> List[NIRSignal]:

    frames = []
    for _ in range(n_frames):
        noisy_voltages = spec.voltages + np.random.normal(
            0.0, noise_scale, size=spec.voltages.shape
        )
        frames.append(NIRSignal(np.clip(noisy_voltages, 0.1, 4.0), spec.wavelengths_nm))

    return frames


def nir_pipeline_from_csv(filename: str) -> NIRFeatures:

    base_spec = load_nir_csv(filename)

    dark_current = np.array([0.02, 0.02, 0.01, 0.03])
    water_reference = np.array([0.15, 0.22, 0.45, 0.88])

    raw_frames = simulate_multi_frame_from_single(base_spec, n_frames=10)

    preprocessed_frames = []
    for f in raw_frames:
        cleaned, _ = preprocess_single_nir_frame(f, dark_current, water_reference)
        preprocessed_frames.append(cleaned)

    acc_spec, frame_var = accumulate_nir_frames(preprocessed_frames)

    acc_cleaned, acc_dbg = preprocess_single_nir_frame(
        acc_spec, dark_current, water_reference
    )
    acc_dbg["frame_variance"] = frame_var

    return extract_nir_features(acc_cleaned, debug_info=acc_dbg)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    nfeats = nir_pipeline_from_csv("synthetic_nir_voltages.csv")

    print("Multi-Wavelength NIR Feature Vector:")
    print(f"Abs 750nm: {nfeats.absorbance_750:.3f}")
    print(f"Abs 810nm: {nfeats.absorbance_810:.3f}")
    print(f"Abs 940nm: {nfeats.absorbance_940:.3f}")
    print(f"Abs 1600nm: {nfeats.absorbance_1600:.3f}")
    print(f"Ratio 1600/940: {nfeats.ratio_1600_to_940:.3f}")
    print(f"Slope: {nfeats.absorption_slope:.4f}")
    print(f"Water index: {nfeats.water_band_index:.3f}")
