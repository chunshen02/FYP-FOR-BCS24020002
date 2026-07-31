"""
explainer.py

Runs local CV forensic heuristics on a face crop already classified as
DEEPFAKE by the BitMind API, producing human-readable evidence strings.
No new pip dependencies: numpy + opencv only.
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

_SIZE = config.FACE_CROP_SIZE
_HANN_2D = np.outer(np.hanning(_SIZE), np.hanning(_SIZE)).astype(np.float32)
_yy, _xx = np.mgrid[0:_SIZE, 0:_SIZE]
_R_NORM = np.sqrt((_yy - _SIZE / 2) ** 2 + (_xx - _SIZE / 2) ** 2) / (_SIZE / 2 * np.sqrt(2))
_LOW_MASK = _R_NORM <= 0.10
_HIGH_MASK = (_R_NORM >= 0.50) & (_R_NORM <= 0.95)
_INNER_SLICE = slice(int(_SIZE * 0.2), int(_SIZE * 0.8))
_SYMMETRY_SLICE = slice(int(_SIZE * 0.1), int(_SIZE * 0.9))


def _fft_high_freq_ratio(gray: np.ndarray) -> float:
    """High/low frequency power ratio; elevated ratio suggests GAN upsampling artifacts."""
    spectrum = np.fft.fftshift(np.fft.fft2(gray * _HANN_2D))
    power = np.abs(spectrum) ** 2
    return float(power[_HIGH_MASK].sum() / (power[_LOW_MASK].sum() + 1e-8))


def _noise_region_ratio(gray: np.ndarray) -> float:
    """Ratio of noise-residual variance between the face interior and its border."""
    residual = gray - cv2.medianBlur(gray, 3)
    border_mask = np.ones((_SIZE, _SIZE), dtype=bool)
    border_mask[_INNER_SLICE, _INNER_SLICE] = False
    inner_var = residual[_INNER_SLICE, _INNER_SLICE].var()
    border_var = residual[border_mask].var()
    lo, hi = sorted((inner_var, border_var))
    return float(hi / (lo + 1e-6))


def _symmetry_deviation(gray: np.ndarray) -> float:
    """Normalized mean absolute difference between the face and its horizontal mirror."""
    diff = np.abs(gray - cv2.flip(gray, 1))
    return float(diff[_SYMMETRY_SLICE, _SYMMETRY_SLICE].mean() / 255.0)


def compute_metrics(face_bgr: np.ndarray) -> dict:
    """Return raw heuristic scores for a face crop, independent of thresholds."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {
        "fft_hf_ratio": _fft_high_freq_ratio(gray),
        "noise_region_ratio": _noise_region_ratio(gray),
        "asymmetry": _symmetry_deviation(gray),
    }


def analyze_artifacts(face_bgr: np.ndarray) -> list:
    """Run local CV artifact heuristics on a DEEPFAKE-classified face crop; return evidence strings."""
    metrics = compute_metrics(face_bgr)
    reasons = []

    if metrics["fft_hf_ratio"] > config.EXPLAIN_FFT_HF_RATIO_THRESHOLD:
        reasons.append(
            "Unusual high-frequency energy in the frequency spectrum, "
            "consistent with GAN upsampling artifacts."
        )
    if metrics["noise_region_ratio"] > config.EXPLAIN_NOISE_RATIO_THRESHOLD:
        reasons.append(
            "Noise pattern differs between the face and its surrounding region, "
            "suggesting the face may have been digitally blended into the frame."
        )
    if metrics["asymmetry"] > config.EXPLAIN_ASYMMETRY_THRESHOLD:
        reasons.append(
            "Left-right facial asymmetry higher than typical, which can occur "
            "when facial features are artificially generated or warped."
        )

    return reasons
