"""
face_thumbnail.py

Small bordered face-crop thumbnail used in Mini Mode, so the user can
visually confirm which specific face a confidence score belongs to
without needing the full video preview.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

from runtime.gui import theme

_RISK_BORDER = {
    "REAL": theme.GREEN,
    "DEEPFAKE": theme.RED,
}
_IMAGE_STYLE = "border: 2px solid {colour}; background-color: #000000;"
_PCT_STYLE = f"color: {{colour}}; font-family: '{theme.MONO_FONT}'; font-size: 10px; font-weight: bold;"


class FaceThumbnail(QWidget):
    """A small cropped-face image with a risk-coloured border and confidence label."""

    def __init__(self, size: int = 64, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._image_label = QLabel()
        self._image_label.setFixedSize(size, size)
        self._image_label.setScaledContents(True)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(_IMAGE_STYLE.format(colour=theme.BORDER))
        layout.addWidget(self._image_label)

        self._pct_label = QLabel("")
        self._pct_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pct_label.setStyleSheet(_PCT_STYLE.format(colour=theme.GREY))
        layout.addWidget(self._pct_label)

        self.setVisible(False)

    def set_face(self, face_bgr: np.ndarray, risk_level: str, p_fake: float):
        """Update the thumbnail with a new BGR crop, verdict-coloured border, and confidence."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._image_label.setPixmap(QPixmap.fromImage(qimg))

        colour = _RISK_BORDER.get(risk_level, theme.GREY)
        self._image_label.setStyleSheet(_IMAGE_STYLE.format(colour=colour))
        self._pct_label.setText(f"{p_fake * 100:.0f}%")
        self._pct_label.setStyleSheet(_PCT_STYLE.format(colour=colour))
        self.setVisible(True)

    def clear(self):
        """Hide this thumbnail slot (fewer faces currently tracked than the pool size)."""
        self.setVisible(False)
