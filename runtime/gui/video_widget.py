"""
video_widget.py

Custom QWidget subclass for displaying annotated video frames with
bounding box overlays, face badges, and LIVE indicator.
"""

import os
import sys
import time
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QFontMetrics
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal, pyqtSlot

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

# Risk-level colour palette (box, badge background, badge text)
_RISK_COLOURS = {
    "REAL":     (QColor("#1D9E75"), QColor("#E1F5EE"), QColor("#085041")),
    "DEEPFAKE": (QColor("#E24B4A"), QColor("#FCEBEB"), QColor("#791F1F")),
}
_PENDING_COLOUR = QColor("#888888")
_PENDING_BADGE_BG = QColor("#2a2a2a")
_PENDING_BADGE_FG = QColor("#cccccc")


class VideoWidget(QWidget):
    """
    Custom widget that displays live video frames with face detection overlays.

    Features:
    - Displays BGR frames emitted by DetectionWorker
    - Draws colour-coded bounding boxes (REAL/DEEPFAKE) around detected faces
    - Adds confidence badges below each box
    - Shows LIVE indicator in top-left corner
    - Scales frames to fit widget maintaining aspect ratio
    - Drag left-click to set a detection zone; right-click to clear
    """

    region_selected = pyqtSignal(object)  # emits (x, y, w, h) in frame pixels, or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("video_widget")
        self.setMinimumHeight(300)
        self.setStyleSheet("background-color: #000000;")

        self._current_pixmap = None
        self._current_results = []
        self._current_pending_bboxes = []
        self._frame_size = (0, 0)
        self._display_size = (0, 0)
        self._offset = (0, 0)

        self._blink_on = True
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)
        self._blink_timer.start(500)

        self._sel_start: QPoint | None = None
        self._sel_end: QPoint | None = None
        self._sel_rect: tuple | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    @pyqtSlot(object, object, object)
    def update_frame(self, frame: np.ndarray, results: list, pending_bboxes: list = None):
        """Update the displayed frame, analyzed results, and pending placeholders."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        self._frame_size = (w, h)

        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        self._current_pixmap = QPixmap.fromImage(qt_image)
        self._current_results = results
        self._current_pending_bboxes = pending_bboxes or []
        self.update()

    def clear(self):
        """Clear the displayed frame."""
        self._current_pixmap = None
        self._current_results = []
        self._current_pending_bboxes = []
        self.update()

    def grab_screenshot(self) -> QPixmap | None:
        """Return a QPixmap of the current frame, or None if no frame is active."""
        return self._current_pixmap

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._current_pixmap is not None:
            self._sel_start = event.position().toPoint()
            self._sel_end = event.position().toPoint()
        elif event.button() == Qt.MouseButton.RightButton:
            self.clear_selection()

    def mouseMoveEvent(self, event):
        if self._sel_start is not None:
            self._sel_end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._sel_start is not None:
            self._sel_end = event.position().toPoint()
            self._commit_selection()
            self._sel_start = None

    def _commit_selection(self):
        if self._display_size[0] == 0 or self._frame_size[0] == 0:
            self._sel_start = None
            return
        ox, oy = self._offset
        dw, dh = self._display_size
        wx1 = max(ox, min(self._sel_start.x(), self._sel_end.x()))
        wy1 = max(oy, min(self._sel_start.y(), self._sel_end.y()))
        wx2 = min(ox + dw, max(self._sel_start.x(), self._sel_end.x()))
        wy2 = min(oy + dh, max(self._sel_start.y(), self._sel_end.y()))
        if wx2 - wx1 < 10 or wy2 - wy1 < 10:
            return
        sx = self._frame_size[0] / self._display_size[0]
        sy = self._frame_size[1] / self._display_size[1]
        self._sel_rect = (
            int((wx1 - ox) * sx),
            int((wy1 - oy) * sy),
            int((wx2 - wx1) * sx),
            int((wy2 - wy1) * sy),
        )
        self.region_selected.emit(self._sel_rect)
        self.update()

    def clear_selection(self):
        """Clear the detection zone selection."""
        self._sel_rect = None
        self._sel_start = None
        self._sel_end = None
        self.region_selected.emit(None)
        self.update()

    @property
    def selection_rect(self) -> tuple | None:
        return self._sel_rect

    def _draw_selection_zone(self, painter: QPainter):
        """Draw the active or in-progress detection zone rectangle."""
        pen = QPen(QColor("#4A9FEF"), 2, Qt.PenStyle.DashLine)
        fill = QColor(74, 159, 239, 30)

        if self._sel_start is not None and self._sel_end is not None:
            rect = QRect(self._sel_start, self._sel_end).normalized()
            painter.setPen(pen)
            painter.setBrush(QBrush(fill))
            painter.drawRect(rect)
            return

        if self._sel_rect is not None and self._display_size[0] > 0:
            fx, fy, fw, fh = self._sel_rect
            sx = self._display_size[0] / self._frame_size[0]
            sy = self._display_size[1] / self._frame_size[1]
            wx = int(fx * sx) + self._offset[0]
            wy = int(fy * sy) + self._offset[1]
            ww = int(fw * sx)
            wh = int(fh * sy)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill))
            painter.drawRect(wx, wy, ww, wh)
            font = QFont("Segoe UI", 9)
            painter.setFont(font)
            painter.setPen(QColor("#4A9FEF"))
            painter.drawText(wx + 4, wy - 4, "Detection Zone")

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
        self._draw_selection_zone(painter)
        self._draw_live_indicator(painter)
        painter.end()

    def _draw_overlays(self, painter: QPainter):
        """Draw colour-coded bounding boxes, confidence badges, and pending placeholders."""
        if self._frame_size[0] == 0:
            return
        if not self._current_results and not self._current_pending_bboxes:
            return

        scale_x = self._display_size[0] / self._frame_size[0]
        scale_y = self._display_size[1] / self._frame_size[1]
        now = time.perf_counter()

        for result in self._current_results:
            if not result.bbox:
                continue

            box_colour, badge_bg, badge_fg = _RISK_COLOURS.get(
                result.risk_level, _RISK_COLOURS["REAL"]
            )

            x = result.bbox["x"]
            y = result.bbox["y"]
            w = result.bbox["w"]
            h = result.bbox["h"]

            scaled_x = int(x * scale_x) + self._offset[0]
            scaled_y = int(y * scale_y) + self._offset[1]
            scaled_w = int(w * scale_x)
            scaled_h = int(h * scale_y)

            painter.setPen(QPen(box_colour, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)

            p_fake_pct = result.p_fake_raw * 100
            elapsed = now - result.last_updated
            age_str = "just now" if elapsed < 1.0 else f"{elapsed:.0f}s ago"
            badge_text = f"{p_fake_pct:.1f}% Deepfake · {age_str}"

            badge_y = scaled_y + scaled_h + 4
            badge_h = 20
            badge_w = self._text_width(badge_text, 10, bold=True) + 10

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

        for bbox in self._current_pending_bboxes:
            if not bbox:
                continue
            scaled_x = int(bbox["x"] * scale_x) + self._offset[0]
            scaled_y = int(bbox["y"] * scale_y) + self._offset[1]
            scaled_w = int(bbox["w"] * scale_x)
            scaled_h = int(bbox["h"] * scale_y)

            pen = QPen(_PENDING_COLOUR, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)

            badge_text = "Analyzing..."
            badge_y = scaled_y + scaled_h + 4
            badge_h = 20
            badge_w = self._text_width(badge_text, 10, bold=True) + 10

            painter.setBrush(QBrush(_PENDING_BADGE_BG))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(scaled_x, badge_y, badge_w, badge_h, 3, 3)

            font = QFont("Segoe UI", 10)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(_PENDING_BADGE_FG)
            painter.drawText(
                scaled_x + 4, badge_y, badge_w - 8, badge_h,
                Qt.AlignmentFlag.AlignCenter,
                badge_text
            )

    def _toggle_blink(self):
        self._blink_on = not self._blink_on
        if self._current_pixmap is not None:
            self.update()

    def _draw_live_indicator(self, painter: QPainter):
        """Draw blinking LIVE indicator in top-left corner of the video area."""
        circle_x = self._offset[0] + 10
        circle_y = self._offset[1] + 10
        circle_size = 10

        if self._blink_on:
            painter.setBrush(QBrush(QColor("#FF3B3B")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(circle_x, circle_y, circle_size, circle_size)

        font = QFont("Segoe UI", 11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 200))
        painter.drawText(
            circle_x + 16, circle_y, 200, circle_size,
            Qt.AlignmentFlag.AlignVCenter,
            "DETECTING"
        )

    @staticmethod
    def _text_width(text: str, font_size: int, bold: bool = False) -> int:
        font = QFont("Segoe UI", font_size)
        font.setBold(bold)
        metrics = QFontMetrics(font)
        return metrics.horizontalAdvance(text)
