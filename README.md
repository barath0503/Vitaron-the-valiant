# Vitaron-the-valiant

VITARON is the world’s first multimodal, AI-powered non-invasive glucose scanner designed to eliminate painful finger-pricks and expensive consumables. By fusing five independent signals—Raman Spectroscopy, Multi-wavelength NIR, Photoacoustic Sensing, PPG, and RF Dielectric Sensing it overcomes traditional barriers like melanin and hydration etc..,

This repository contains a research prototype implementation that demonstrates:
->Sensor-level signal preprocessing pipelines
->Feature extraction from each modality
->Physics-informed quality metrics
->Rule-based multimodal fusion architecture

The current fusion layer is deterministic and model-driven.
A machine learning model has not yet been integrated due to the absence of a clinically validated multimodal dataset.

The feature relationships and baseline data used in this prototype are derived from:
->Publicly available laboratory research
->Academic publications
->Synthetic signal simulations

This codebase is intended for:
->Architectural demonstration
->Signal processing and validation
->Multimodal fusion understanding

## What was upgraded

- Replaced synthetic PAS CSV with lab source-data points from a peer-reviewed paper.
- Replaced synthetic RF sweep with lab-derived dielectric reactance sweep using published fitted equations.
- Added `true_glucose_mg_dl` labels to all modality CSV files.
- Refactored PAS and RF pipelines to load lab schemas robustly.
- Made inference deterministic with seeded random generators.
- Removed confidence inflation and added explicit predicted-vs-true comparison metrics.

## Data provenance used

1. PAS lab data:
   - Nature Metabolism (2024), DOI: `10.1038/s42255-024-01016-9`
   - Source data file (Fig. 4d):
     `https://static-content.springer.com/esm/art%3A10.1038%2Fs42255-024-01016-9/MediaObjects/42255_2024_1016_MOESM5_ESM.xlsx`

2. RF lab-derived sweep:
   - Omprakash et al. (2020), TSI Journal, DOI: `10.37532/tsij.2020.14(4).178`
   - Equation used from Table 2 at 150 mg/dL: `X = b + m*log10(f)` with `b=63535.90`, `m=-18565.6`

## Ground-truth convention

All CSV files contain a `true_glucose_mg_dl` label. Current reference value is `150 mg/dL`.

## Run full fusion and compare with true glucose

```bash
python Fusion_product/ensemble_product.py
```

The script prints:
- glucose estimate
- confidence and uncertainty
- safety status
- true glucose
- absolute error and ARD (%)
=======


