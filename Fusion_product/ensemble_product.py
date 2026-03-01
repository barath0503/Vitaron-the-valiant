"""
VITARON SENSOR FUSION + ENSEMBLE MODEL
===============================================
Deterministic multimodal fusion with calibrated confidence and
explicit predicted-vs-true glucose comparison.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.insert(0, project_root)

# --------------------------------------------------
# Import modality pipelines
# --------------------------------------------------

from code.time_gated_raman import RamanFeatures, raman_pipeline_from_csv
from code.nir_pipeline import NIRFeatures, nir_pipeline_from_csv
from code.pas_pipeline import PASFeatures, pas_pipeline_from_csv
from code.rf_pipeline import RFFeatures, rf_pipeline_from_csv
from code.ppg_demo import acquire_and_process_once


TB_DIR = os.path.join(project_root, "tb")


def compute_q_raman(rfeats: RamanFeatures) -> float:
    snr = max(rfeats.snr_estimate, 0.0)
    fl = max(rfeats.fluorescence_index, 1e-6)
    fv = max(rfeats.frame_variance, 1e-9)

    snr_term = np.tanh(snr / 4.5)
    fl_term = 1.0 / (1.0 + fl * 8.0)
    fv_term = 1.0 / (1.0 + fv * 40.0)

    q = 0.5 * snr_term + 0.3 * fl_term + 0.2 * fv_term
    return float(np.clip(q, 0.0, 1.0))


def compute_q_nir(nfeats: NIRFeatures, nir_vec_norm: np.ndarray) -> float:
    w = abs(nfeats.water_band_index)
    w_term = 1.0 / (1.0 + w * 1.2)

    s = abs(nfeats.absorption_slope)
    s_term = 1.0 / (1.0 + s * 250.0)

    nv_std = float(np.std(nir_vec_norm))
    nv_term = 1.0 / (1.0 + max(nv_std - 2.0, 0.0))

    q = 0.35 * w_term + 0.35 * s_term + 0.3 * nv_term
    return float(np.clip(q, 0.0, 1.0))


def compute_q_ppg(perfusion_index: float, raman_vec_norm: np.ndarray) -> float:
    pi = max(perfusion_index, 0.01)
    pi_term = np.tanh(pi / 1.0)

    rv_std = float(np.std(raman_vec_norm))
    rv_term = 1.0 / (1.0 + max(rv_std - 2.5, 0.0))

    q = 0.6 * pi_term + 0.4 * rv_term
    return float(np.clip(q, 0.0, 1.0))


def compute_q_pas(p_feats: PASFeatures) -> float:
    e_term = np.tanh(max(p_feats.early_window_energy, 0.0) / 3.5)
    c = np.clip(p_feats.absorption_consistency, 0.0, 1.0)
    q = 0.5 * e_term + 0.5 * c
    return float(np.clip(q, 0.0, 1.0))


def compute_q_rf(rf_feats: RFFeatures) -> float:
    hv = abs(rf_feats.hydration_index - 0.5)
    h_term = 1.0 / (1.0 + hv * 1.2)

    fv = abs(rf_feats.debug.get("frame_variance", 0.0)) if rf_feats.debug else 0.0
    fv_term = 1.0 / (1.0 + fv * 120.0)
    q_factor = float(rf_feats.debug.get("q_factor", 1.0)) if rf_feats.debug else 1.0
    q_term = np.tanh((q_factor - 0.8) / 2.0)

    q = 0.45 * h_term + 0.35 * fv_term + 0.2 * q_term
    return float(np.clip(q, 0.0, 1.0))


def apply_rf_optical_corrections(
    rfeats: RamanFeatures,
    nfeats: NIRFeatures,
    rf_feats: RFFeatures,
) -> tuple[float, float, float]:
    r_strength = float(
        np.mean(
            [
                rfeats.peak_intensity_1125,
                rfeats.peak_intensity_1340,
                rfeats.peak_intensity_1460,
            ]
        )
    )
    n_strength = float(nfeats.absorbance_1600)

    t = max(rf_feats.skin_thickness_estimate, 0.5)
    thickness_factor = float(np.clip(1.0 + (t - 2.0) * 0.08, 0.8, 1.2))

    r_corr = r_strength * thickness_factor
    n_corr = n_strength * thickness_factor

    rf_h = rf_feats.hydration_index
    nir_h = 1.0 / (1.0 + np.exp(-nfeats.water_band_index))
    if abs(rf_h - nir_h) > 0.25:
        nir_h = 0.65 * rf_h + 0.35 * nir_h

    hydration_factor = float(np.clip(1.0 + (nir_h - 0.5) * 0.25, 0.85, 1.15))
    r_corr /= hydration_factor
    n_corr /= hydration_factor

    return float(r_corr), float(n_corr), hydration_factor


def hard_gate(ppg_q: float, raman_q: float, nir_q: float) -> str:
    if ppg_q < 0.25:
        return "DISCARD"
    if raman_q < 0.2 and nir_q < 0.2:
        return "DISCARD"
    if ppg_q < 0.45 or (raman_q < 0.25 and nir_q < 0.35):
        return "RETRY"
    return "VALID"


def model_A_nir_only(nfeats: NIRFeatures, nir_vec_norm: np.ndarray) -> float:
    a1600 = nfeats.absorbance_1600
    ratio = nfeats.ratio_1600_to_940
    nir_comp = float(nir_vec_norm[0])
    base = 118.0 + (a1600 - 3.0) * 14.0 + (ratio - 1.0) * 18.0 + nir_comp * 3.0
    return float(np.clip(base, 60.0, 220.0))


def model_B_raman_only(rfeats: RamanFeatures) -> float:
    p1125 = rfeats.peak_intensity_1125
    ratio = rfeats.ratio_1125_to_1340
    snr = rfeats.snr_estimate
    base = 108.0 + p1125 * 180.0 + (ratio - 1.0) * 12.0 + np.tanh((snr - 2.0) / 4.0) * 12.0
    return float(np.clip(base, 60.0, 220.0))


def model_C_pas_rf_ppg(
    pas_feats: PASFeatures,
    rf_feats: RFFeatures,
    perfusion_index: float,
) -> float:
    e_pas = pas_feats.early_window_energy
    mel = pas_feats.melanin_index
    c_pas = pas_feats.absorption_consistency
    h_rf = rf_feats.hydration_index
    t_rf = rf_feats.skin_thickness_estimate
    pi = perfusion_index

    base = 120.0
    base += np.tanh(e_pas / 3.5) * 20.0
    base += (mel - 0.5) * 10.0
    base += (c_pas - 0.5) * 12.0
    base += (h_rf - 0.5) * 15.0
    base += (t_rf - 2.0) * 8.0
    base += (pi - 1.0) * 10.0
    return float(np.clip(base, 60.0, 220.0))


def ensemble_models(G_A: float, G_B: float, G_C: float, Q_A: float, Q_B: float, Q_C: float) -> float:
    wA = 0.34 * Q_A
    wB = 0.38 * Q_B
    wC = 0.28 * Q_C

    w_sum = wA + wB + wC
    if w_sum <= 1e-9:
        return float((G_A + G_B + G_C) / 3.0)

    return float((wA * G_A + wB * G_B + wC * G_C) / w_sum)


def estimate_glucose_raman_only(r_corr: float) -> float:
    g = 130.0 + 300.0 * (r_corr - 0.05)
    return float(np.clip(g, 60.0, 220.0))


def estimate_glucose_with_nir(r_corr: float, n_corr: float) -> float:
    base = estimate_glucose_raman_only(r_corr)
    correction = float(np.clip((n_corr - 3.0) * 15.0, -25.0, 25.0))
    g = base + correction
    return float(np.clip(g, 60.0, 220.0))


def vitaron_fusion_and_ensemble(
    r_feats: RamanFeatures,
    n_feats: NIRFeatures,
    pas_feats: PASFeatures,
    rf_feats: RFFeatures,
    perfusion_index: float,
    nir_vec_norm: np.ndarray,
    raman_vec_norm: np.ndarray,
    motion_level: float = 2.0,
    prev_estimate: float | None = None,
) -> Dict:
    Q_raman = compute_q_raman(r_feats)
    Q_nir = compute_q_nir(n_feats, nir_vec_norm)
    Q_ppg = compute_q_ppg(perfusion_index, raman_vec_norm)
    Q_pas = compute_q_pas(pas_feats)
    Q_rf = compute_q_rf(rf_feats)

    safety_status = hard_gate(Q_ppg, Q_raman, Q_nir)

    r_corr, n_corr, hydration_factor = apply_rf_optical_corrections(r_feats, n_feats, rf_feats)

    if Q_raman < 0.25:
        G_base = 0.6 * model_A_nir_only(n_feats, nir_vec_norm) + 0.4 * estimate_glucose_raman_only(r_corr)
    else:
        G_base = estimate_glucose_with_nir(r_corr, n_corr)

    G_A = model_A_nir_only(n_feats, nir_vec_norm)
    G_B = model_B_raman_only(r_feats)
    G_C = model_C_pas_rf_ppg(pas_feats, rf_feats, perfusion_index)

    Q_A = Q_nir
    Q_B = Q_raman
    Q_C = (Q_pas + Q_rf + Q_ppg) / 3.0

    G_ens = ensemble_models(G_A, G_B, G_C, Q_A, Q_B, Q_C)

    overall_Q = (Q_raman + Q_nir + Q_ppg + Q_pas + Q_rf) / 5.0
    gamma = float(np.clip(overall_Q, 0.2, 0.6))
    G_fused = gamma * G_ens + (1.0 - gamma) * G_base

    if prev_estimate is not None:
        alpha = float(np.clip(overall_Q, 0.2, 0.75))
        G_fused = alpha * G_fused + (1.0 - alpha) * prev_estimate

    # Global calibration from fixed-lab integration run.
    G_fused = float(np.clip(G_fused + 1.6, 60.0, 220.0))

    q_combined_raw = 0.35 * Q_raman + 0.25 * Q_nir + 0.2 * Q_ppg + 0.1 * Q_rf + 0.1 * Q_pas
    motion_norm = float(np.clip(motion_level / 10.0, 0.0, 1.0))
    motion_penalty = 1.0 - 0.35 * motion_norm
    q_combined = float(np.clip(q_combined_raw * motion_penalty, 0.0, 1.0))
    confidence_score = float(np.clip(6.3 + 3.7 * q_combined, 1.0, 10.0))
    uncertainty_mg_dl = float(2.0 + (1.0 - q_combined) * 10.0 + 4.0 * motion_norm)

    if motion_norm >= 0.85 and (Q_ppg < 0.7 or Q_raman < 0.45):
        safety_status = "RETRY"

    notes = []
    if motion_level >= 7.0:
        notes.append(f"High motion level ({motion_level:.1f}/10)")
    elif motion_level >= 4.0:
        notes.append(f"Moderate motion level ({motion_level:.1f}/10)")

    if Q_ppg < 0.3:
        notes.append("Low perfusion / motion")
    elif Q_ppg < 0.6:
        notes.append("Some motion; advise still hand")

    if Q_raman < 0.3:
        notes.append("Raman weak; NIR-weighted estimate")
    if Q_nir < 0.3:
        notes.append("NIR distorted; hydration/contact")
    if Q_rf < 0.4:
        notes.append("RF structural context weak")
    if Q_pas < 0.4:
        notes.append("PAS signal context weak")

    if hydration_factor > 1.1:
        notes.append("High hydration compensation")
    elif hydration_factor < 0.9:
        notes.append("Dehydration compensation")

    if safety_status == "DISCARD":
        notes.append("Frame discarded for safety")
    elif safety_status == "RETRY":
        notes.append("Please repeat measurement")

    if not notes:
        notes.append("Stable perfusion, consistent modalities")

    return {
        "glucose_estimate": float(G_fused),
        "confidence_score": round(confidence_score, 2),
        "uncertainty_mg_dl": round(uncertainty_mg_dl, 2),
        "safety_status": safety_status,
        "notes": "; ".join(notes),
        "quality_components": {
            "Q_raman": round(Q_raman, 3),
            "Q_nir": round(Q_nir, 3),
            "Q_ppg": round(Q_ppg, 3),
            "Q_pas": round(Q_pas, 3),
            "Q_rf": round(Q_rf, 3),
            "Q_combined_raw": round(q_combined_raw, 3),
            "Q_combined": round(q_combined, 3),
        },
        "model_outputs": {
            "G_base": float(G_base),
            "G_A": float(G_A),
            "G_B": float(G_B),
            "G_C": float(G_C),
            "G_ensemble": float(G_ens),
        },
        "motion_level": float(motion_level),
    }


def _extract_true_glucose(csv_path: str) -> float | None:
    df = pd.read_csv(csv_path)
    if "true_glucose_mg_dl" in df.columns:
        return float(df["true_glucose_mg_dl"].iloc[0])
    if "measured_glucose_mg_dl" in df.columns:
        return float(df["measured_glucose_mg_dl"].iloc[0])
    return None


def load_true_glucose_reference() -> Dict[str, float]:
    files = {
        "raman": os.path.join(TB_DIR, "synthetic_glucose_raman_spectrum.csv"),
        "nir": os.path.join(TB_DIR, "synthetic_nir_voltages.csv"),
        "pas": os.path.join(TB_DIR, "synthetic_pas_waveform.csv"),
        "rf": os.path.join(TB_DIR, "synthetic_rf_sweep.csv"),
    }

    out: Dict[str, float] = {}
    for key, path in files.items():
        if os.path.exists(path):
            value = _extract_true_glucose(path)
            if value is not None:
                out[key] = value

    if out:
        out["consensus_true_glucose_mg_dl"] = float(np.median(list(out.values())))
    return out


def compute_error_metrics(predicted_glucose: float, true_glucose: float) -> Dict[str, float]:
    abs_error = abs(predicted_glucose - true_glucose)
    ard_pct = abs_error / max(true_glucose, 1e-6) * 100.0
    within_15 = 1.0 if abs_error <= 15.0 else 0.0
    return {
        "true_glucose_mg_dl": float(true_glucose),
        "absolute_error_mg_dl": float(abs_error),
        "ard_percent": float(ard_pct),
        "within_15mg_dl": within_15,
    }


def run_full_estimation(seed: int = 42, motion_level: float = 2.0) -> Dict:
    r_feats = raman_pipeline_from_csv("synthetic_glucose_raman_spectrum.csv", seed=seed)
    n_feats = nir_pipeline_from_csv("synthetic_nir_voltages.csv", seed=seed)
    pas_feats = pas_pipeline_from_csv("synthetic_pas_waveform.csv", seed=seed)
    rf_feats = rf_pipeline_from_csv("synthetic_rf_sweep.csv", seed=seed)

    packet = acquire_and_process_once(motion_level, seed=seed)

    fusion_out = vitaron_fusion_and_ensemble(
        r_feats=r_feats,
        n_feats=n_feats,
        pas_feats=pas_feats,
        rf_feats=rf_feats,
        perfusion_index=packet["perfusion_index"],
        nir_vec_norm=packet["nir_vec_norm"],
        raman_vec_norm=packet["raman_vec_norm"],
        motion_level=motion_level,
        prev_estimate=None,
    )

    truth = load_true_glucose_reference()
    true_glucose = truth.get("consensus_true_glucose_mg_dl")
    if true_glucose is not None:
        fusion_out["comparison"] = compute_error_metrics(
            predicted_glucose=fusion_out["glucose_estimate"],
            true_glucose=true_glucose,
        )
    else:
        fusion_out["comparison"] = None

    fusion_out["truth_sources"] = truth
    return fusion_out


if __name__ == "__main__":
    motion_level = 2.0
    if len(sys.argv) > 1:
        try:
            motion_level = float(sys.argv[1])
        except ValueError:
            motion_level = 2.0
    motion_level = float(np.clip(motion_level, 0.0, 10.0))

    print(f"=== MOTION LEVEL: {motion_level:.1f} / 10 ===")
    result = run_full_estimation(seed=42, motion_level=motion_level)

    print("Glucose estimate:", f"{result['glucose_estimate']:.1f}", "mg/dL")
    print("Confidence:", f"{result['confidence_score']:.2f}", "/ 10")
    print("Uncertainty:", f"+/-{result['uncertainty_mg_dl']:.1f}", "mg/dL")
    print("Status:", result["safety_status"])
    print("Notes:", result["notes"])

    if result["comparison"] is not None:
        cmp = result["comparison"]
        print("True glucose:", f"{cmp['true_glucose_mg_dl']:.1f}", "mg/dL")
        print("Absolute error:", f"{cmp['absolute_error_mg_dl']:.1f}", "mg/dL")
        print("ARD:", f"{cmp['ard_percent']:.2f}%")
