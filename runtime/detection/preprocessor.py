"""
preprocessor.py

Converts a BGR face crop into a normalised float32 tensor
ready for ONNX Runtime inference.
"""

import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class Preprocessor:
    """
    Prepares a BGR face crop for EfficientNet-B4 ONNX inference.
    Preprocessing is identical to the training pipeline to prevent
    train-inference distribution mismatch.
    """

    def __init__(self):
        self._mean = np.array(config.INPUT_MEAN, dtype=np.float32)
        self._std = np.array(config.INPUT_STD, dtype=np.float32)
        self._size = (config.INPUT_SIZE, config.INPUT_SIZE)

    def process(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a BGR face crop to a float32 inference tensor.

        Args:
            face_bgr: BGR numpy array (any size).

        Returns:
            float32 numpy array of shape (1, 3, 224, 224).
        """
        # Step 1: Resize using bicubic interpolation
        image = cv2.resize(face_bgr, self._size, interpolation=cv2.INTER_CUBIC)

        # Step 2: BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 3: Normalise to [0, 1] then apply ImageNet mean/std
        image = image.astype(np.float32) / 255.0
        image = (image - self._mean) / self._std

        # Step 4: HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # Step 5: Add batch dimension -> (1, 3, 224, 224)
        image = np.expand_dims(image, axis=0)

        return image.astype(np.float32)
