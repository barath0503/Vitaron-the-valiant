"""
PAS (Photoacoustic Spectroscopy) Processing Pipeline
==================================================
Stages:
1. Raw waveform load (from tb CSV)
2. Baseline subtraction + denoising
3. Early-window isolation
4. Feature extraction for fusion
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from scipy.signal import savgol_filter


# =============================================================================
# PATH RESOLUTION (NEW)
# =============================================================================

def resolve_csv_path(filename: str) -> str:
    """
    Structure:
    VITARON-THE-VALIANT/
        code/
            pas_pipeline.py
        tb/
            synthetic_pas_waveform.csv
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))   # code/
    project_root = os.path.dirname(current_dir)                # repo root
    tb_folder = os.path.join(project_root, "tb")

    return os.path.join(tb_folder, filename)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PASWaveform:
    time_us: np.ndarray
    voltage: np.ndarray
    metadata: Dict | None = None


@dataclass
class PASFeatures:
    early_peak_amplitude: float
    early_window_energy: float
    absorption_consistency: float
    melanin_index: float
    debug: Dict | None = None


# =============================================================================
# LOADING
# =============================================================================

def load_pas_csv(filename: str) -> PASWaveform:
    csv_path = resolve_csv_path(filename)
    df = pd.read_csv(csv_path)

    return PASWaveform(
        time_us=df["time_us"].to_numpy(),
        voltage=df["voltage_raw"].to_numpy(),
        metadata={"source": "csv", "path": csv_path},
    )


# =============================================================================
# PROCESSING UTILITIES
# =============================================================================

def baseline_subtract_pas(waveform: PASWaveform, pre_trigger_us: float = 15.0):

    pre_mask = waveform.time_us < pre_trigger_us
    baseline_level = float(np.mean(waveform.voltage[pre_mask]))

    corrected_voltage = waveform.voltage - baseline_level

    return (
        PASWaveform(
            waveform.time_us.copy(),
            corrected_voltage,
            waveform.metadata,
        ),
        baseline_level,
    )


def isolate_early_window(waveform: PASWaveform,
                         start_us: float = 15.0,
                         end_us: float = 50.0):

    mask = (waveform.time_us >= start_us) & (waveform.time_us <= end_us)
    return waveform.time_us[mask], waveform.voltage[mask]


# =============================================================================
# FRAME PROCESSING
# =============================================================================

def preprocess_single_pas_frame(waveform: PASWaveform):

    corrected, baseline_level = baseline_subtract_pas(waveform)

    # Denoising
    if len(corrected.voltage) > 21:
        corrected.voltage = savgol_filter(corrected.voltage, 21, 3)

    debug = {
        "baseline_level": baseline_level,
        "frame_variance": 0.0,
    }

    return corrected, debug


def accumulate_pas_frames(frames: List[PASWaveform]):

    voltages = np.stack([f.voltage for f in frames], axis=0)
    mean_voltage = np.mean(voltages, axis=0)
    frame_var = float(np.mean(np.var(voltages, axis=0)))

    return (
        PASWaveform(frames[0].time_us.copy(), mean_voltage),
        frame_var,
    )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_pas_features(waveform: PASWaveform, debug_info=None):

    early_time, early_voltage = isolate_early_window(waveform)

    if len(early_voltage) == 0:
        return PASFeatures(0.0, 0.0, 0.0, 0.0, {})

    early_peak_amplitude = float(np.max(np.abs(early_voltage)))
    early_window_energy = float(np.sum(early_voltage ** 2))

    # Melanin proxy (normalized)
    melanin_index = float(np.clip(early_peak_amplitude / 0.8, 0.0, 1.0))

    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info else 0.0
    consistency = float(np.clip(1.0 - frame_var / 0.1, 0.0, 1.0))

    return PASFeatures(
        early_peak_amplitude,
        early_window_energy,
        consistency,
        melanin_index,
        debug={
            "early_peak_time": float(
                early_time[np.argmax(np.abs(early_voltage))]
            ),
            "frame_variance": frame_var,
        },
    )


# =============================================================================
# MAIN PIPELINE ENTRY
# =============================================================================

def pas_pipeline_from_csv(filename: str):

    base_waveform = load_pas_csv(filename)

    raw_frames = [
        PASWaveform(
            base_waveform.time_us,
            base_waveform.voltage + np.random.normal(0, 0.01, size=base_waveform.voltage.shape),
        )
        for _ in range(8)
    ]

    processed = []
    for f in raw_frames:
        cleaned, _ = preprocess_single_pas_frame(f)
        processed.append(cleaned)

    acc_waveform, frame_var = accumulate_pas_frames(processed)

    acc_cleaned, dbg = preprocess_single_pas_frame(acc_waveform)
    dbg["frame_variance"] = frame_var

    return extract_pas_features(acc_cleaned, dbg)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":

    pas_feats = pas_pipeline_from_csv("synthetic_pas_waveform.csv")

    print("PAS (Photoacoustic) Feature Vector:")
    print(f"Early peak: {pas_feats.early_peak_amplitude:.3f} V")
    print(f"Early energy: {pas_feats.early_window_energy:.3f}")
    print(f"Consistency: {pas_feats.absorption_consistency:.3f}")
    print(f"Melanin index: {pas_feats.melanin_index:.3f}")
