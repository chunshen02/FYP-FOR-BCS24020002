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

    def process(self, face_bgr: np.ndarray) -> bytes:
        """
        Convert a BGR face crop to JPEG bytes.

        Args:
            face_bgr: BGR numpy array of the aligned face crop (any size).

        Returns:
            JPEG-encoded bytes of the face crop.
        """
        size = config.FACE_CROP_SIZE
        resized = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(
            '.jpg', resized,
            [cv2.IMWRITE_JPEG_QUALITY, config.FACE_CROP_JPEG_QUALITY]
        )
        return buf.tobytes()
