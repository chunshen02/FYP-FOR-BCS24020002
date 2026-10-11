"""
preprocessor.py

Prepares a BGR face crop for submission to the BitMind API.
Outputs JPEG-encoded bytes ready for an HTTP POST request.
"""

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class Preprocessor:
    """
    Converts a BGR face crop to JPEG bytes for BitMind API submission.

    cv2.imencode handles BGR→YCbCr internally, so no channel swap is needed.
    The resulting JPEG decodes to correct RGB on the server side.
    """

    def __init__(self):
        self._gamma_lut = np.array(
            [min(255, int((i / 255.0) ** config.LOW_LIGHT_GAMMA * 255)) for i in range(256)],
            dtype=np.uint8,
        )

    def _prepare(self, face_bgr: np.ndarray) -> np.ndarray:
        """Resize and gamma-correct a BGR face crop to FACE_CROP_SIZE."""
        size = config.FACE_CROP_SIZE
        # Cubic shrinking aliases fine texture, which could read as a synthesis artifact to the classifier.
        interpolation = cv2.INTER_AREA if min(face_bgr.shape[:2]) > size else cv2.INTER_CUBIC
        resized = cv2.resize(face_bgr, (size, size), interpolation=interpolation)
        if cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).mean() < config.LOW_LIGHT_THRESHOLD:
            # Gamma correction brightens dark crops so the API receives a cleaner image.
            resized = cv2.LUT(resized, self._gamma_lut)
        return resized

    def process_with_array(self, face_bgr: np.ndarray) -> tuple:
        """
        Convert a BGR face crop to JPEG bytes, also returning the pre-encode
        BGR array for local artifact analysis (avoids re-decoding the JPEG,
        which would contaminate frequency/noise measurements with compression
        artifacts).

        Returns:
            (jpeg_bytes, resized_bgr)
        """
        resized = self._prepare(face_bgr)
        _, buf = cv2.imencode(
            '.jpg', resized,
            [cv2.IMWRITE_JPEG_QUALITY, config.FACE_CROP_JPEG_QUALITY]
        )
        return buf.tobytes(), resized
