"""
inference.py

Runs EfficientNet-B4 ONNX inference on face crops and returns
raw and EMA-smoothed fake confidence scores.
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


@dataclass
class FaceResult:
    """Holds inference results for a single detected face."""

    p_fake_raw: float  # Raw P_fake - used for alert triggering
    p_fake_display: float  # EMA-smoothed P_fake - used for visual overlay
    risk_level: str  # "GREEN", "AMBER", or "RED"
    bbox: dict = None  # Optional: {"x": int, "y": int, "w": int, "h": int} for visual overlays


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def _get_risk_level(p_fake: float) -> str:
    if p_fake >= config.FAKE_THRESHOLD:
        return "RED"
    if p_fake >= config.WARNING_THRESHOLD:
        return "AMBER"
    return "GREEN"


def _get_best_providers() -> list:
    """
    Automatically select the best available ONNX Runtime execution providers.
    Priority: CUDA (NVIDIA GPU) > DirectML (Windows GPU) > Azure > CPU
    """
    available = ort.get_available_providers()
    print(f"[INFO] Available ONNX providers: {available}")

    priority = [
        "CUDAExecutionProvider",
        "DirectMLExecutionProvider",
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]

    selected = [p for p in priority if p in available]
    print(f"[INFO] Selected providers: {selected}")
    return selected


class DeepfakeInferenceEngine:
    """
    Loads the EfficientNet-B4 ONNX model and runs per-face inference.
    Maintains per-face EMA state for smoothed display scores.
    """

    def __init__(self):
        if not os.path.exists(config.ONNX_MODEL_PATH):
            raise FileNotFoundError(
                f"ONNX model not found at: {config.ONNX_MODEL_PATH}\n"
                "Run training/model/export_onnx.py first."
            )

        self._session = ort.InferenceSession(
            config.ONNX_MODEL_PATH,
            providers=_get_best_providers(),
        )
        self._input_name = self._session.get_inputs()[0].name

        # EMA state: one smoothed value per face slot (indexed by face order in frame)
        self._ema_state: list = []

    def run(self, tensors: list) -> list:
        """
        Run inference on a list of preprocessed face tensors.

        Args:
            tensors: List of float32 numpy arrays, each shape (1, 3, 224, 224).
                     One tensor per detected face in the current frame.

        Returns:
            List of FaceResult objects, one per face.
        """
        results = []

        # Extend EMA state if more faces appear than previously tracked
        while len(self._ema_state) < len(tensors):
            self._ema_state.append(0.0)

        for i, tensor in enumerate(tensors):
            logits = self._session.run(None, {self._input_name: tensor})[0][0]
            probs = _softmax(logits)
            p_fake_raw = float(probs[1])

            # EMA smoothing for display only
            prev_ema = self._ema_state[i]
            p_fake_display = config.EMA_ALPHA * p_fake_raw + (1 - config.EMA_ALPHA) * prev_ema
            self._ema_state[i] = p_fake_display

            results.append(
                FaceResult(
                    p_fake_raw=p_fake_raw,
                    p_fake_display=p_fake_display,
                    risk_level=_get_risk_level(p_fake_raw),  # raw for risk level
                )
            )

        # Reset EMA slots for faces that disappeared this frame
        if len(tensors) < len(self._ema_state):
            self._ema_state = self._ema_state[: len(tensors)]

        return results

    def reset_ema(self):
        """Reset all EMA state (call when detection is stopped and restarted)."""
        self._ema_state = []
