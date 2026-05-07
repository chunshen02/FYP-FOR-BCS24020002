"""
video_widget.py

Custom QWidget subclass for displaying annotated video frames with
bounding box overlays, face badges, and LIVE indicator.
"""

import os
import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QFontMetrics
)
from PyQt6.QtCore import Qt, pyqtSlot

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

# Risk-level colour palette (box, badge background, badge text)
_RISK_COLOURS = {
    "GREEN": (QColor("#1D9E75"), QColor("#E1F5EE"), QColor("#085041")),
    "AMBER": (QColor("#EF9F27"), QColor("#FAEEDA"), QColor("#412402")),
    "RED":   (QColor("#E24B4A"), QColor("#FCEBEB"), QColor("#791F1F")),
}


class VideoWidget(QWidget):
    """
    Custom widget that displays live video frames with face detection overlays.

    Features:
    - Displays BGR frames emitted by DetectionWorker
    - Draws colour-coded bounding boxes (GREEN/AMBER/RED) around detected faces
    - Adds confidence badges below each box
    - Shows LIVE indicator in top-left corner
    - Scales frames to fit widget maintaining aspect ratio
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("video_widget")
        self.setMinimumHeight(300)
        self.setStyleSheet("background-color: #000000;")

        self._current_pixmap = None
        self._current_results = []
        self._frame_size = (0, 0)
        self._display_size = (0, 0)
        self._offset = (0, 0)

    @pyqtSlot(object, object)
    def update_frame(self, frame: np.ndarray, results: list):
        """Update the displayed frame and detection results."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        self._frame_size = (w, h)

        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        self._current_pixmap = QPixmap.fromImage(qt_image)
        self._current_results = results
        self.update()

    def clear(self):
        """Clear the displayed frame."""
        self._current_pixmap = None
        self._current_results = []
        self.update()

    def grab_screenshot(self) -> QPixmap | None:
        """Return a QPixmap of the current frame, or None if no frame is active."""
        return self._current_pixmap

    def paintEvent(self, event):
        painter = QPainter(self)

        if self._current_pixmap is None:
            painter.fillRect(self.rect(), QColor("#000000"))
            font = QFont("Segoe UI", 13)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255, 77))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No feed — press Start to begin detection."
            )
            painter.end()
            return

        widget_size = self.size()
        scaled_pixmap = self._current_pixmap.scaled(
            widget_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        x_offset = (widget_size.width() - scaled_pixmap.width()) // 2
        y_offset = (widget_size.height() - scaled_pixmap.height()) // 2
        self._display_size = (scaled_pixmap.width(), scaled_pixmap.height())
        self._offset = (x_offset, y_offset)

        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        self._draw_overlays(painter)
        self._draw_live_indicator(painter)
        painter.end()

    def _draw_overlays(self, painter: QPainter):
        """Draw colour-coded bounding boxes and confidence badges for detected faces."""
        if not self._current_results or self._frame_size[0] == 0:
            return

        scale_x = self._display_size[0] / self._frame_size[0]
        scale_y = self._display_size[1] / self._frame_size[1]

        for result in self._current_results:
            if not result.bbox:
                continue

            box_colour, badge_bg, badge_fg = _RISK_COLOURS.get(
                result.risk_level, _RISK_COLOURS["GREEN"]
            )

            x = result.bbox["x"]
            y = result.bbox["y"]
            w = result.bbox["w"]
            h = result.bbox["h"]

            scaled_x = int(x * scale_x) + self._offset[0]
            scaled_y = int(y * scale_y) + self._offset[1]
            scaled_w = int(w * scale_x)
            scaled_h = int(h * scale_y)

            # Bounding box
            painter.setPen(QPen(box_colour, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)

            # Badge text — use smoothed score for display
            confidence = result.p_fake_display * 100
            if result.p_fake_display >= config.WARNING_THRESHOLD:
                badge_text = f"Deepfake {confidence:.1f}%"
            else:
                badge_text = f"Real {(1 - result.p_fake_display) * 100:.1f}%"

            badge_y = scaled_y + scaled_h + 4
            badge_h = 20
            badge_w = self._text_width(badge_text, 10) + 10

            painter.setBrush(QBrush(badge_bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(scaled_x, badge_y, badge_w, badge_h, 3, 3)

            font = QFont("Segoe UI", 10)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(badge_fg)
            painter.drawText(
                scaled_x + 4, badge_y, badge_w - 8, badge_h,
                Qt.AlignmentFlag.AlignCenter,
                badge_text
            )

    def _draw_live_indicator(self, painter: QPainter):
        """Draw LIVE indicator in top-left corner of the video area."""
        circle_x = self._offset[0] + 8
        circle_y = self._offset[1] + 8
        circle_size = 8

        painter.setBrush(QBrush(QColor("#00FF00")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(circle_x, circle_y, circle_size, circle_size)

        font = QFont("Segoe UI", 11)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 127))
        painter.drawText(
            circle_x + 12, circle_y, 200, 16,
            Qt.AlignmentFlag.AlignVCenter,
            "LIVE — Zoom Meeting"
        )

    @staticmethod
    def _text_width(text: str, font_size: int) -> int:
        font = QFont("Segoe UI", font_size)
        metrics = QFontMetrics(font)
        return metrics.horizontalAdvance(text)
