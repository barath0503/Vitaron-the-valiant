"""
PAS (Photoacoustic Spectroscopy) Processing Pipeline
==================================================
Consumes lab PAS data from CSV and extracts robust features for fusion.
Supports:
1. Legacy waveform schema: time_us, voltage_raw
2. Lab delta schema: sample_index, delta_area_pct, true_glucose_mg_dl
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def resolve_csv_path(filename: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    tb_folder = os.path.join(project_root, "tb")
    return os.path.join(tb_folder, filename)


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


def load_pas_csv(filename: str) -> PASWaveform:
    csv_path = resolve_csv_path(filename)
    df = pd.read_csv(csv_path)

    metadata: Dict = {"source": "csv", "path": csv_path}
    if "true_glucose_mg_dl" in df.columns:
        metadata["true_glucose_mg_dl"] = float(df["true_glucose_mg_dl"].iloc[0])
    if "source" in df.columns:
        metadata["lab_source"] = str(df["source"].iloc[0])

    if {"time_us", "voltage_raw"}.issubset(df.columns):
        return PASWaveform(
            time_us=df["time_us"].to_numpy(dtype=float),
            voltage=df["voltage_raw"].to_numpy(dtype=float),
            metadata=metadata,
        )

    # Lab delta schema from published source data
    if {"sample_index", "delta_area_pct"}.issubset(df.columns):
        return PASWaveform(
            time_us=df["sample_index"].to_numpy(dtype=float),
            voltage=df["delta_area_pct"].to_numpy(dtype=float),
            metadata=metadata,
        )

    raise ValueError(
        "Unsupported PAS CSV schema. Expected either "
        "{time_us, voltage_raw} or {sample_index, delta_area_pct}."
    )


def baseline_subtract_pas(waveform: PASWaveform) -> Tuple[PASWaveform, float]:
    n = max(1, int(0.1 * len(waveform.voltage)))
    baseline_level = float(np.mean(waveform.voltage[:n]))
    corrected_voltage = waveform.voltage - baseline_level

    return (
        PASWaveform(
            waveform.time_us.copy(),
            corrected_voltage,
            waveform.metadata,
        ),
        baseline_level,
    )


def isolate_early_window(
    waveform: PASWaveform, early_fraction: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    cutoff = int(max(2, np.floor(len(waveform.time_us) * early_fraction)))
    return waveform.time_us[:cutoff], waveform.voltage[:cutoff]


def preprocess_single_pas_frame(waveform: PASWaveform) -> Tuple[PASWaveform, Dict]:
    corrected, baseline_level = baseline_subtract_pas(waveform)

    if len(corrected.voltage) >= 7:
        window = min(15, len(corrected.voltage) - (1 - len(corrected.voltage) % 2))
        window = max(5, window)
        if window % 2 == 0:
            window -= 1
        corrected.voltage = savgol_filter(corrected.voltage, window, 2)

    debug = {
        "baseline_level": baseline_level,
        "frame_variance": 0.0,
    }
    return corrected, debug


def accumulate_pas_frames(frames: List[PASWaveform]) -> Tuple[PASWaveform, float]:
    voltages = np.stack([f.voltage for f in frames], axis=0)
    mean_voltage = np.mean(voltages, axis=0)
    frame_var = float(np.mean(np.var(voltages, axis=0)))

    return (
        PASWaveform(frames[0].time_us.copy(), mean_voltage, frames[0].metadata),
        frame_var,
    )


def extract_pas_features(waveform: PASWaveform, debug_info: Dict | None = None) -> PASFeatures:
    early_time, early_voltage = isolate_early_window(waveform)
    if len(early_voltage) == 0:
        return PASFeatures(0.0, 0.0, 0.0, 0.0, {})

    early_peak_amplitude = float(np.max(np.abs(early_voltage)))
    early_window_energy = float(np.mean(early_voltage**2))

    signal_std = float(np.std(waveform.voltage))
    signal_mean_abs = float(np.mean(np.abs(waveform.voltage)) + 1e-6)
    cv = signal_std / signal_mean_abs
    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info else 0.0
    consistency = float(np.clip(1.0 / (1.0 + cv + frame_var), 0.0, 1.0))

    spread = float(np.percentile(np.abs(waveform.voltage), 95) - np.percentile(np.abs(waveform.voltage), 50))
    melanin_index = float(np.clip(spread / (spread + 5.0), 0.0, 1.0))

    debug = {
        "early_peak_time": float(early_time[np.argmax(np.abs(early_voltage))]),
        "frame_variance": frame_var,
    }
    if waveform.metadata and "true_glucose_mg_dl" in waveform.metadata:
        debug["true_glucose_mg_dl"] = waveform.metadata["true_glucose_mg_dl"]
    if waveform.metadata and "lab_source" in waveform.metadata:
        debug["lab_source"] = waveform.metadata["lab_source"]

    return PASFeatures(
        early_peak_amplitude=early_peak_amplitude,
        early_window_energy=early_window_energy,
        absorption_consistency=consistency,
        melanin_index=melanin_index,
        debug=debug,
    )


def pas_pipeline_from_csv(filename: str, seed: int = 42, n_frames: int = 6) -> PASFeatures:
    base_waveform = load_pas_csv(filename)
    rng = np.random.default_rng(seed)

    raw_frames = []
    for _ in range(n_frames):
        noise = rng.normal(0.0, 0.03, size=base_waveform.voltage.shape)
        raw_frames.append(
            PASWaveform(
                base_waveform.time_us,
                base_waveform.voltage + noise,
                metadata=base_waveform.metadata,
            )
        )

    processed = []
    for frame in raw_frames:
        cleaned, _ = preprocess_single_pas_frame(frame)
        processed.append(cleaned)

    acc_waveform, frame_var = accumulate_pas_frames(processed)
    acc_cleaned, dbg = preprocess_single_pas_frame(acc_waveform)
    dbg["frame_variance"] = frame_var

    return extract_pas_features(acc_cleaned, dbg)


if __name__ == "__main__":
    pas_feats = pas_pipeline_from_csv("synthetic_pas_waveform.csv")
    print("PAS Feature Vector:")
    print(f"Early peak: {pas_feats.early_peak_amplitude:.3f}")
    print(f"Early energy: {pas_feats.early_window_energy:.3f}")
    print(f"Consistency: {pas_feats.absorption_consistency:.3f}")
    print(f"Melanin index: {pas_feats.melanin_index:.3f}")
