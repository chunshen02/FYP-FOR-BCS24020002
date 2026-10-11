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

# Risk-level colour palette: box outline, solid pill background, pill text
_RISK_COLOURS = {
    "REAL":     (QColor("#22C55E"), QColor("#22C55E"), QColor("#052E16")),
    "DEEPFAKE": (QColor("#EF4444"), QColor("#EF4444"), QColor("#2D0A0A")),
}
_PENDING_COLOUR = QColor("#6B7280")
_PENDING_BADGE_BG = QColor("#12151C")
_PENDING_BADGE_FG = QColor("#9CA3AF")
_RECHECK_COLOUR = QColor("#F59E0B")
_RECHECK_BUTTON_BG = QColor("#12151C")
_RECHECK_BUTTON_FG = QColor("#9CA3AF")

# Badge: bold percentage on top, smaller risk label + timestamp beneath.
# Badges are drawn at one of these size tiers -- full size unless it would
# overlap a neighboring face's badge, in which case progressively smaller
# tiers are tried so crowded faces still each get a legible badge instead
# of overlapping ones (see VideoWidget._place_badge).
_BADGE_SCALE_TIERS = (1.0, 0.8, 0.65, 0.5)


def _tiered_fonts(base_pt: int) -> dict:
    fonts = {}
    for scale in _BADGE_SCALE_TIERS:
        f = QFont("Segoe UI", max(int(round(base_pt * scale)), 6))
        f.setBold(True)
        fonts[scale] = f
    return fonts


_BADGE_PCT_FONTS = _tiered_fonts(15)
_BADGE_PCT_METRICS = {s: QFontMetrics(f) for s, f in _BADGE_PCT_FONTS.items()}

_BADGE_SUB_FONTS = _tiered_fonts(8)
_BADGE_SUB_METRICS = {s: QFontMetrics(f) for s, f in _BADGE_SUB_FONTS.items()}

_BADGE_FONTS = _tiered_fonts(10)
_BADGE_METRICS = {s: QFontMetrics(f) for s, f in _BADGE_FONTS.items()}

_LIVE_FONT = QFont("Segoe UI", 11)
_LIVE_FONT.setBold(True)


def _fit_pixmap(frame_bgr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    """Scale a BGR frame to fit max_w x max_h (aspect ratio kept) and convert it to a QPixmap."""
    h, w = frame_bgr.shape[:2]
    k = min(max_w / w, max_h / h)
    size = (max(1, int(w * k)), max(1, int(h * k)))
    scaled = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    return QPixmap.fromImage(QImage(scaled.data, size[0], size[1], 3 * size[0], QImage.Format.Format_BGR888))


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
    recheck_requested = pyqtSignal(str)   # emits the face_id whose "Re-check" button was clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("video_widget")
        self.setMinimumHeight(300)
        self.setStyleSheet("background-color: #000000;")

        self._current_frame = None
        self._current_results = []
        self._current_pending_bboxes = []
        self._current_rechecking_bboxes = []
        self._recheck_hitboxes = []  # list[(QRect, face_id)] in widget coords, rebuilt every paint
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

    @pyqtSlot(object, object, object, object)
    def update_frame(self, frame: np.ndarray, results: list, pending_bboxes: list,
                      rechecking_results: list):
        """Update the displayed frame, analyzed results, and pending/rechecking placeholders."""
<<<<<<< HEAD
        # Hidden in Mini mode — skip the per-frame resize and QPixmap conversion.
        if not self.isVisible():
            return
        h, w = frame.shape[:2]
        self._frame_size = (w, h)

        # Downscaling once here with OpenCV is far cheaper than converting the
        # full-resolution frame and smooth-scaling it in Qt on every paint.
        scale = min(self.width() / w, self.height() / h)
        display_w, display_h = max(1, int(w * scale)), max(1, int(h * scale))
        display = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
        qt_image = QImage(display.data, display_w, display_h, 3 * display_w, QImage.Format.Format_BGR888)
        self._current_pixmap = QPixmap.fromImage(qt_image)
=======
        # Converted to a pixmap in paintEvent at display size only (~1.4 ms vs ~7 ms for converting
        # the full 2560x1600 frame here), and frames Qt never paints cost nothing.
        self._current_frame = frame
        self._frame_size = (frame.shape[1], frame.shape[0])
        self._cached_scaled_pixmap = None  # invalidate on new frame
>>>>>>> 3bef4a09fe8c51c26d29703668d4889ad5fe4d31
        self._current_results = results
        self._current_pending_bboxes = pending_bboxes
        self._current_rechecking_bboxes = [r.bbox for r in rechecking_results]
        self.update()

    def clear(self):
        """Clear the displayed frame."""
<<<<<<< HEAD
        self._current_pixmap = None
=======
        self._current_frame = None
        self._cached_scaled_pixmap = None
>>>>>>> 3bef4a09fe8c51c26d29703668d4889ad5fe4d31
        self._current_results = []
        self._current_pending_bboxes = []
        self._current_rechecking_bboxes = []
        self._recheck_hitboxes = []
        self.update()

    @property
    def has_frame(self) -> bool:
        """Whether a live frame is currently displayed."""
        return self._current_frame is not None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._current_frame is not None:
            pos = event.position().toPoint()
            for rect, face_id in self._recheck_hitboxes:
                if rect.contains(pos):
                    self.recheck_requested.emit(face_id)
                    return
            self._sel_start = pos
            self._sel_end = pos
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

        if self._current_frame is None:
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

<<<<<<< HEAD
        # The pixmap is already display-sized by update_frame; after a widget
        # resize it catches up on the next frame (~66 ms at DISPLAY_FPS=15).
        pixmap = self._current_pixmap
        x_offset = (self.width() - pixmap.width()) // 2
        y_offset = (self.height() - pixmap.height()) // 2
        self._display_size = (pixmap.width(), pixmap.height())
=======
        widget_size = self.size()
        current_w_size = (widget_size.width(), widget_size.height())
        if self._cached_scaled_pixmap is None or self._cached_widget_size != current_w_size:
            self._cached_scaled_pixmap = _fit_pixmap(self._current_frame, *current_w_size)
            self._cached_widget_size = current_w_size
        scaled_pixmap = self._cached_scaled_pixmap

        x_offset = (widget_size.width() - scaled_pixmap.width()) // 2
        y_offset = (widget_size.height() - scaled_pixmap.height()) // 2
        self._display_size = (scaled_pixmap.width(), scaled_pixmap.height())
>>>>>>> 3bef4a09fe8c51c26d29703668d4889ad5fe4d31
        self._offset = (x_offset, y_offset)

        painter.drawPixmap(x_offset, y_offset, pixmap)
        self._draw_overlays(painter)
        self._draw_selection_zone(painter)
        self._draw_live_indicator(painter)
        painter.end()

    def _draw_overlays(self, painter: QPainter):
        """Draw bounding boxes, then collision-aware badges (results, then pending/rechecking)."""
        self._recheck_hitboxes = []
        if self._frame_size[0] == 0:
            return
        if (not self._current_results and not self._current_pending_bboxes
                and not self._current_rechecking_bboxes):
            return

        scale_x = self._display_size[0] / self._frame_size[0]
        scale_y = self._display_size[1] / self._frame_size[1]
        now = time.perf_counter()

        # Bounding boxes reflect actual detected face position, so they are
        # never moved or resized -- only badge placement is collision-aware.
        for result in self._current_results:
            if not result.bbox:
                continue
            box_colour, _, _ = _RISK_COLOURS.get(result.risk_level, _RISK_COLOURS["REAL"])
            painter.setPen(QPen(box_colour, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self._scaled_rect(result.bbox, scale_x, scale_y), 6, 6)

        for bbox in self._current_pending_bboxes:
            self._draw_bbox_outline(painter, bbox, scale_x, scale_y, _PENDING_COLOUR)
        for bbox in self._current_rechecking_bboxes:
            self._draw_bbox_outline(painter, bbox, scale_x, scale_y, _RECHECK_COLOUR)

        # Badges: results first (highest priority — an actual verdict),
        # then pending, then rechecking, each avoiding every badge already
        # placed this frame.
        placed_badges: list = []
        for result in self._current_results:
            if not result.bbox:
                continue
            self._draw_result_badge(painter, result, scale_x, scale_y, now, placed_badges)
        for bbox in self._current_pending_bboxes:
            if bbox:
                self._draw_placeholder_badge(painter, bbox, scale_x, scale_y, "Analyzing...",
                                              _PENDING_COLOUR, _PENDING_BADGE_FG, placed_badges)
        for bbox in self._current_rechecking_bboxes:
            if bbox:
                self._draw_placeholder_badge(painter, bbox, scale_x, scale_y, "Re-checking...",
                                              _RECHECK_COLOUR, _RECHECK_COLOUR, placed_badges)

    def _scaled_rect(self, bbox: dict, scale_x: float, scale_y: float) -> QRect:
        """Map a frame-pixel bbox dict to a widget-coordinate QRect."""
        return QRect(
            int(bbox["x"] * scale_x) + self._offset[0],
            int(bbox["y"] * scale_y) + self._offset[1],
            int(bbox["w"] * scale_x),
            int(bbox["h"] * scale_y),
        )

    def _draw_bbox_outline(self, painter: QPainter, bbox: dict, scale_x: float, scale_y: float,
                            colour: QColor):
        """Dashed outline only, for a face with no current result (pending or rechecking)."""
        if not bbox:
            return
        painter.setPen(QPen(colour, 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self._scaled_rect(bbox, scale_x, scale_y))

    def _place_badge(self, box_rect: QRect, measure_fn, placed: list) -> tuple:
        """
        Find a non-overlapping placement for a badge anchored to box_rect.

        measure_fn(scale) -> (width, height) returns the badge's footprint
        at a given size tier (using that tier's own font metrics, so
        shrinking actually shrinks the rendered text, not just an empty
        box). Tries every combination of {below, above} x {left-anchored,
        right-anchored, centered} at each tier from full size down to the
        smallest -- horizontal offsets matter as much as vertical ones when
        neighboring faces are side by side, not just stacked. If nothing
        avoids overlap, forces the smallest/first placement so a badge is
        never simply dropped.

        Returns (rect, scale).
        """
        widget_w = self.width()
        fallback = None
        for scale in _BADGE_SCALE_TIERS:
            w, h = measure_fn(scale)
            x_candidates = (box_rect.left(), box_rect.right() - w, box_rect.center().x() - w // 2)
            for y in (box_rect.bottom() + 6, box_rect.top() - h - 6):
                for x in x_candidates:
                    x = max(0, min(x, widget_w - w)) if widget_w > w else max(0, x)
                    rect = QRect(x, max(0, y), w, h)
                    if fallback is None:
                        fallback = (rect, scale)
                    if not any(rect.intersects(p) for p in placed):
                        return rect, scale
        return fallback

    def _draw_result_badge(self, painter: QPainter, result, scale_x: float, scale_y: float,
                            now: float, placed: list):
        _, badge_bg, badge_fg = _RISK_COLOURS.get(result.risk_level, _RISK_COLOURS["REAL"])
        box_rect = self._scaled_rect(result.bbox, scale_x, scale_y)

        p_fake_pct = result.p_fake_raw * 100
        elapsed = now - result.last_updated
        age_str = "just now" if elapsed < 1.0 else f"{elapsed:.0f}s ago"
        pct_text = f"{p_fake_pct:.1f}%"
        sub_text = f"{result.risk_level.upper()}  ·  {age_str}"
        recheck_label = "↻ Re-check"

        # The Re-check button rides along beside the badge, so its footprint
        # must be part of the same collision search -- otherwise the badge
        # itself never overlaps a neighbor, but the button tacked on next to
        # it can still poke into one.
        def measure(scale):
            badge_w = max(
                _BADGE_PCT_METRICS[scale].horizontalAdvance(pct_text),
                _BADGE_SUB_METRICS[scale].horizontalAdvance(sub_text),
            ) + int(20 * scale)
            h = int(46 * scale)
            if not result.face_id:
                return badge_w, h
            button_w = _BADGE_METRICS[scale].horizontalAdvance(recheck_label) + int(16 * scale)
            return badge_w + int(6 * scale) + button_w, h

        combined_rect, scale = self._place_badge(box_rect, measure, placed)
        placed.append(combined_rect)

        badge_w = max(
            _BADGE_PCT_METRICS[scale].horizontalAdvance(pct_text),
            _BADGE_SUB_METRICS[scale].horizontalAdvance(sub_text),
        ) + int(20 * scale)
        rect = QRect(combined_rect.x(), combined_rect.y(), badge_w, combined_rect.height())

        painter.setBrush(QBrush(badge_bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, int(10 * scale), int(10 * scale))

        painter.setFont(_BADGE_PCT_FONTS[scale])
        painter.setPen(badge_fg)
        painter.drawText(
            rect.x(), rect.y() + int(3 * scale), rect.width(), int(24 * scale),
            Qt.AlignmentFlag.AlignCenter,
            pct_text
        )

        painter.setFont(_BADGE_SUB_FONTS[scale])
        fg = QColor(badge_fg)
        fg.setAlpha(200)
        painter.setPen(fg)
        painter.drawText(
            rect.x(), rect.y() + int(27 * scale), rect.width(), int(16 * scale),
            Qt.AlignmentFlag.AlignCenter,
            sub_text
        )

        if result.face_id:
            button_rect = QRect(
                rect.right() + int(6 * scale), rect.y(),
                combined_rect.right() - rect.right() - int(6 * scale), rect.height(),
            )
            self._draw_recheck_button(painter, button_rect, scale, recheck_label, result.face_id)

    def _draw_placeholder_badge(self, painter: QPainter, bbox: dict, scale_x: float, scale_y: float,
                                 label: str, box_colour: QColor, text_colour: QColor, placed: list):
        box_rect = self._scaled_rect(bbox, scale_x, scale_y)

        def measure(scale):
            w = _BADGE_METRICS[scale].horizontalAdvance(label) + int(10 * scale)
            h = int(20 * scale)
            return w, h

        rect, scale = self._place_badge(box_rect, measure, placed)
        placed.append(rect)

        painter.setBrush(QBrush(_PENDING_BADGE_BG))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 3, 3)

        painter.setFont(_BADGE_FONTS[scale])
        painter.setPen(text_colour)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_recheck_button(self, painter: QPainter, rect: QRect, scale: float,
                              label: str, face_id: str):
        """Draw the 'Re-check' pill at an already-resolved rect; records its hitbox for mousePressEvent."""
        painter.setBrush(QBrush(_RECHECK_BUTTON_BG))
        painter.setPen(QPen(_PENDING_COLOUR, 1))
        painter.drawRoundedRect(rect, int(10 * scale), int(10 * scale))

        painter.setFont(_BADGE_FONTS[scale])
        painter.setPen(_RECHECK_BUTTON_FG)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        self._recheck_hitboxes.append((rect, face_id))

    def _toggle_blink(self):
        self._blink_on = not self._blink_on
        if self._current_frame is not None:
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

        painter.setFont(_LIVE_FONT)
        painter.setPen(QColor(255, 255, 255, 200))
        painter.drawText(
            circle_x + 16, circle_y, 200, circle_size,
            Qt.AlignmentFlag.AlignVCenter,
            "DETECTING"
        )

