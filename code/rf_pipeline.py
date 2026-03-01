"""
RF Dielectric Sensing Pipeline
=========================================
Consumes lab dielectric sweep data and extracts stable RF context features.
Supported schema:
1. Lab sweep: frequency_hz, reactance_ohm, true_glucose_mg_dl
2. Legacy schema fallback: frequency_mhz, s11_magnitude_db, s11_phase_deg
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def resolve_csv_path(filename: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    tb_folder = os.path.join(project_root, "tb")
    return os.path.join(tb_folder, filename)


@dataclass
class RFSweep:
    frequency_hz: np.ndarray
    reactance_ohm: np.ndarray
    metadata: Dict | None = None


@dataclass
class RFFeatures:
    effective_permittivity: float
    phase_delay: float
    hydration_index: float
    skin_thickness_estimate: float
    debug: Dict | None = None


def load_rf_csv(filename: str) -> RFSweep:
    csv_path = resolve_csv_path(filename)
    df = pd.read_csv(csv_path)

    metadata: Dict = {"source": "csv", "path": csv_path}
    if "true_glucose_mg_dl" in df.columns:
        metadata["true_glucose_mg_dl"] = float(df["true_glucose_mg_dl"].iloc[0])
    if "source" in df.columns:
        metadata["lab_source"] = str(df["source"].iloc[0])

    if {"frequency_hz", "reactance_ohm"}.issubset(df.columns):
        return RFSweep(
            frequency_hz=df["frequency_hz"].to_numpy(dtype=float),
            reactance_ohm=df["reactance_ohm"].to_numpy(dtype=float),
            metadata=metadata,
        )

    # Backward compatibility with legacy sweep schema.
    if {"frequency_mhz", "s11_magnitude_db"}.issubset(df.columns):
        freq_hz = df["frequency_mhz"].to_numpy(dtype=float) * 1e6
        reactance_ohm = np.abs(df["s11_magnitude_db"].to_numpy(dtype=float)) * 100.0
        return RFSweep(
            frequency_hz=freq_hz,
            reactance_ohm=reactance_ohm,
            metadata=metadata,
        )

    raise ValueError(
        "Unsupported RF CSV schema. Expected either "
        "{frequency_hz, reactance_ohm} or {frequency_mhz, s11_magnitude_db}."
    )


def find_resonance_feature(sweep: RFSweep) -> Tuple[float, float]:
    freq_log = np.log10(np.clip(sweep.frequency_hz, 1.0, None))
    grad = np.gradient(sweep.reactance_ohm, freq_log)
    peak_idx = int(np.argmin(grad))
    f_res = float(sweep.frequency_hz[peak_idx])

    # Robust shape-quality proxy from gradient concentration.
    abs_grad = np.abs(grad)
    p50 = float(np.percentile(abs_grad, 50))
    p95 = float(np.percentile(abs_grad, 95))
    q_like = float(np.clip(p95 / (p50 + 1e-6), 1.0, 12.0))
    return f_res, q_like


def preprocess_rf_sweep(sweep: RFSweep) -> Tuple[RFSweep, Dict]:
    order = np.argsort(sweep.frequency_hz)
    freq_sorted = sweep.frequency_hz[order]
    react_sorted = sweep.reactance_ohm[order]

    window = min(15, max(5, len(react_sorted) // 12))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    react_smooth = np.convolve(react_sorted, kernel, mode="same")

    cleaned = RFSweep(freq_sorted, react_smooth, sweep.metadata)
    return cleaned, {"raw_resonance": find_resonance_feature(cleaned)}


def accumulate_rf_frames(frames: List[RFSweep]) -> Tuple[RFSweep, float]:
    reacts = np.stack([f.reactance_ohm for f in frames], axis=0)
    mean_react = np.mean(reacts, axis=0)
    frame_var_abs = float(np.mean(np.var(reacts, axis=0)))
    signal_scale = float(np.mean(np.abs(mean_react)) + 1e-6)
    frame_var = float(np.sqrt(frame_var_abs) / signal_scale)

    return (
        RFSweep(frames[0].frequency_hz.copy(), mean_react, frames[0].metadata),
        frame_var,
    )


def extract_rf_features(sweep: RFSweep, debug_info: Dict | None = None) -> RFFeatures:
    f_res_hz, q_factor = find_resonance_feature(sweep)
    react = sweep.reactance_ohm

    react_norm = (react - np.min(react)) / (np.ptp(react) + 1e-9)
    mid_idx = len(react_norm) // 2
    effective_permittivity = float(25.0 + 45.0 * (1.0 - react_norm[mid_idx]))

    phase_delay = float(np.std(react))
    hydration_index = float(np.clip(np.mean(react_norm[-20:]), 0.0, 1.0))

    freq_log = np.log10(np.clip(sweep.frequency_hz, 1.0, None))
    slope = float(np.polyfit(freq_log, react, 1)[0])
    skin_thickness = float(np.clip(1.2 + abs(slope) / 20000.0, 0.8, 3.5))

    frame_var = float(debug_info.get("frame_variance", 0.0)) if debug_info else 0.0
    debug = {
        "resonance_hz": f_res_hz,
        "q_factor": q_factor,
        "frame_variance": frame_var,
    }
    if sweep.metadata and "true_glucose_mg_dl" in sweep.metadata:
        debug["true_glucose_mg_dl"] = sweep.metadata["true_glucose_mg_dl"]
    if sweep.metadata and "lab_source" in sweep.metadata:
        debug["lab_source"] = sweep.metadata["lab_source"]

    return RFFeatures(
        effective_permittivity=effective_permittivity,
        phase_delay=phase_delay,
        hydration_index=hydration_index,
        skin_thickness_estimate=skin_thickness,
        debug=debug,
    )


def rf_pipeline_from_csv(filename: str, seed: int = 42, n_frames: int = 6) -> RFFeatures:
    base_sweep = load_rf_csv(filename)
    rng = np.random.default_rng(seed)

    raw_frames = []
    for _ in range(n_frames):
        noise = rng.normal(0.0, 12.0, size=base_sweep.reactance_ohm.shape)
        raw_frames.append(
            RFSweep(
                base_sweep.frequency_hz,
                base_sweep.reactance_ohm + noise,
                metadata=base_sweep.metadata,
            )
        )

    processed = []
    for frame in raw_frames:
        cleaned, _ = preprocess_rf_sweep(frame)
        processed.append(cleaned)

    acc_sweep, frame_var = accumulate_rf_frames(processed)
    acc_cleaned, dbg = preprocess_rf_sweep(acc_sweep)
    dbg["frame_variance"] = frame_var

    return extract_rf_features(acc_cleaned, dbg)


if __name__ == "__main__":
    rf_feats = rf_pipeline_from_csv("synthetic_rf_sweep.csv")
    print("RF Feature Vector:")
    print(f"Permittivity: {rf_feats.effective_permittivity:.2f}")
    print(f"Phase-delay proxy: {rf_feats.phase_delay:.2f}")
    print(f"Hydration index: {rf_feats.hydration_index:.3f}")
    print(f"Skin thickness: {rf_feats.skin_thickness_estimate:.2f} mm")
