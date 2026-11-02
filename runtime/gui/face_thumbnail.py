"""
face_thumbnail.py

Small bordered face-crop thumbnail used in Mini Mode, so the user can
visually confirm which specific face a confidence score belongs to
without needing the full video preview, plus the per-face verdict
tooltip shared with the video preview and the on-screen face locator.
"""

import html

import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal

from runtime.gui import theme

_RISK_BORDER = {
    "REAL": theme.GREEN,
    "DEEPFAKE": theme.RED,
}
_IMAGE_STYLE = "border: 2px solid {colour}; background-color: #000000;"
_PCT_STYLE = f"color: {{colour}}; font-family: '{theme.MONO_FONT}'; font-size: 10px; font-weight: bold;"
_THUMBNAIL_HINT = "Click: show on screen · Shift+click: re-check"

NO_ARTIFACTS_REASON = (
    "No local visual artifacts detected by heuristic analysis — "
    "this result is based on the cloud model's own assessment."
)


def face_details_html(result, hint: str) -> str:
    """Tooltip for one face: its verdict and score, the reasons behind it, then an action hint."""
    colour = _RISK_BORDER.get(result.risk_level, theme.GREY)
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in result.explanation or [NO_ARTIFACTS_REASON])
    return (
        f"<b style='color: {colour};'>{result.risk_level} · {result.p_fake_raw * 100:.1f}%</b>"
        f"<ul style='margin-left: -20px;'>{reasons}</ul>"
        f"<span style='color: {theme.TEXT_DIM};'>{hint}</span>"
    )


class FaceThumbnail(QWidget):
    """A small cropped-face image with a risk-coloured border and confidence label; click to locate, Shift+click to re-check."""

    locate_requested = pyqtSignal(str)    # face_id to highlight on screen
    recheck_requested = pyqtSignal(str)   # face_id to re-check

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

        self._face_id = ""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setVisible(False)

    def set_face(self, result, rechecking: bool):
        """Show a face's analyzed crop with a verdict-coloured border (amber while re-checking) and its confidence."""
        self._face_id = result.face_id
        rgb = cv2.cvtColor(result.thumbnail, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._image_label.setPixmap(QPixmap.fromImage(qimg))

        colour = theme.AMBER if rechecking else _RISK_BORDER.get(result.risk_level, theme.GREY)
        self._image_label.setStyleSheet(_IMAGE_STYLE.format(colour=colour))
        self._pct_label.setText(f"{result.p_fake_raw * 100:.0f}%")
        self._pct_label.setStyleSheet(_PCT_STYLE.format(colour=colour))
        self.setToolTip(face_details_html(result, "Re-checking…" if rechecking else _THUMBNAIL_HINT))
        self.setVisible(True)

    def clear(self):
        """Hide this thumbnail slot (fewer faces currently tracked than the pool size)."""
        self._face_id = ""
        self.setVisible(False)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._face_id:
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.recheck_requested.emit(self._face_id)
        else:
            self.locate_requested.emit(self._face_id)
