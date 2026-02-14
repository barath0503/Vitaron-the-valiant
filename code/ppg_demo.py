"""
PPG + Motion + Optical Vector Simulation Module
===============================================
Provides normalized vectors for fusion layer.
Designed for integration with VITARON-THE-VALIANT fusion engine.
"""

from __future__ import annotations

import time
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

TRUE_GLUCOSE = 115.0  # mg/dL (Demo ground truth)


# =============================================================================
# SENSOR RAW SIMULATION
# =============================================================================

def get_nir_raw_stream(motion_level: float) -> np.ndarray:
    """
    Simulates 4-wavelength NIR voltages.
    Motion introduces lift artifact + noise.
    """
    base_voltages = np.array([3.42, 3.15, 2.88, 1.25])

    noise_amp = motion_level / 20.0
    noise = np.random.normal(0, noise_amp, 4)

    lift = 0.0
    if motion_level > 5:
        lift = np.random.uniform(0.1, 0.5)

    return base_voltages - lift + noise


def get_raman_raw_features(motion_level: float) -> np.ndarray:
    """
    Simulates Raman peak intensities (1125, 1340, 1460).
    Motion reduces signal strength + increases noise.
    """
    base_peaks = np.array([0.915, 0.726, 0.537])

    attenuation = 1.0 - (motion_level / 15.0)
    attenuation = max(attenuation, 0.1)

    noise = np.random.normal(0, 0.02 * motion_level, 3)

    return (base_peaks * attenuation) + noise


def get_ppg_perfusion_index(motion_level: float) -> float:
    """
    Simulates PPG perfusion index (PI).
    Motion creates false spikes or dropouts.
    """
    true_pi = 1.2

    if motion_level > 3:
        artifact = np.random.uniform(-0.5, 0.5) * (motion_level / 5.0)
        return max(0.1, true_pi + artifact)

    return true_pi + np.random.normal(0, 0.05)


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def process_nir_vector(raw_voltages: np.ndarray):
    """
    1. Hydration compensation (940 nm reference)
    2. SNV normalization
    """
    water_ref = 2.88
    correction_factor = raw_voltages[2] / water_ref
    correction_factor = max(correction_factor, 0.1)

    corrected = raw_voltages / correction_factor

    std = np.std(corrected)
    if std < 1e-6:
        std = 1e-6

    normalized = (corrected - np.mean(corrected)) / std

    return normalized, correction_factor


def process_raman_vector(raw_peaks: np.ndarray, ppg_pi: float):
    """
    Blood volume normalization.
    """
    ppg_pi = max(ppg_pi, 0.2)
    return raw_peaks / ppg_pi


# =============================================================================
# ACQUISITION ENTRY POINT (USED BY FUSION ENGINE)
# =============================================================================

def acquire_and_process_once(motion_level: float):
    """
    Main entry point for fusion layer.
    Returns normalized vectors + SQI.
    """

    # --- RAW ---
    nir_raw = get_nir_raw_stream(motion_level)
    raman_raw = get_raman_raw_features(motion_level)
    perfusion_index = get_ppg_perfusion_index(motion_level)

    # --- PROCESS ---
    nir_vec_norm, nir_correction = process_nir_vector(nir_raw)
    raman_vec_norm = process_raman_vector(raman_raw, perfusion_index)

    # --- SIMPLE QUALITY METRICS ---
    nir_variance = float(np.var(nir_raw))
    raman_strength = float(np.mean(raman_raw))

    return {
        "perfusion_index": float(perfusion_index),
        "nir_vec_norm": nir_vec_norm,
        "raman_vec_norm": raman_vec_norm,
        "nir_variance": nir_variance,
        "raman_strength": raman_strength,
        "nir_correction_factor": float(nir_correction),
    }


# =============================================================================
# OPTIONAL TERMINAL DEMO
# =============================================================================

def run_terminal_demo():
    while True:
        user_input = input("\nEnter Motion Level (0-10) [q to quit]: ")

        if user_input.lower() == "q":
            break

        try:
            motion_level = float(user_input)
            motion_level = np.clip(motion_level, 0, 10)

            packet = acquire_and_process_once(motion_level)

            print("\n--- SENSOR TELEMETRY ---")
            print(f"Perfusion Index: {packet['perfusion_index']:.3f}")
            print(f"NIR Norm Vector: {np.array2string(packet['nir_vec_norm'], precision=3)}")
            print(f"Raman Norm Vector: {np.array2string(packet['raman_vec_norm'], precision=3)}")
            print(f"NIR Variance: {packet['nir_variance']:.4f}")
            print(f"Raman Strength: {packet['raman_strength']:.4f}")
            print("-" * 50)

        except ValueError:
            print("Invalid input. Enter 0–10.")


if __name__ == "__main__":
    run_terminal_demo()
