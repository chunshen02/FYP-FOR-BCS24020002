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
from PyQt6.QtWidgets import QToolTip, QWidget
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QFontMetrics, QLinearGradient
)
from PyQt6.QtCore import Qt, QEvent, QTimer, QPoint, QPointF, QRect, QRectF, pyqtSignal, pyqtSlot

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from runtime.gui import theme

# Risk-level colour: box brackets, badge border and badge text
_RISK_COLOURS = {
    "REAL":     QColor(theme.GREEN),
    "DEEPFAKE": QColor(theme.RED),
}
_BADGE_BG = QColor(theme.SURFACE)
_BADGE_BG.setAlpha(225)
_PENDING_COLOUR = QColor(theme.GREY)
_PENDING_BADGE_FG = QColor(theme.TEXT_DIM)
_RECHECK_COLOUR = QColor(theme.AMBER)
_RECHECK_BUTTON_FG = QColor(theme.CYAN)
_ZONE_COLOUR = QColor(theme.MAGENTA)
_FRAME_BRACKET_COLOUR = QColor(theme.CYAN)
_GRID_COLOUR = QColor(theme.GRID)
_GRID_STEP = 28

# Animations below are timed with time.perf_counter() and drawn in the repaints a running
# feed already triggers (one per frame at DISPLAY_FPS), so they add no repaints of their own.
_LOCK_S = 0.4          # brackets close in on a face this long after its verdict arrives or changes
_LOCK_SPREAD = 14      # px outside the box where the closing brackets start
_SCAN_PERIOD_S = 3.5   # one pass of the scan band down the live frame
_SCAN_BAND = 40        # px height of the scan band

# The idle screen has no feed driving repaints, so its scan line has a timer that runs
# only while that screen is visible, and repaints only the strips the line moves through.
_IDLE_TICK_MS = 33
_IDLE_PERIOD_S = 4.0
_IDLE_TRAIL = 60       # px of fading trail above the idle scan line


def _cyan(alpha: int) -> QColor:
    colour = QColor(theme.CYAN)
    colour.setAlpha(alpha)
    return colour


_SCAN_CLEAR = _cyan(0)
_SCAN_BAND_PEAK = _cyan(26)
_IDLE_TRAIL_PEAK = _cyan(40)
_IDLE_LINE = _cyan(150)

# Badge: bold percentage on top, smaller risk label + timestamp beneath.
# Badges are drawn at one of these size tiers -- full size unless it would
# overlap a neighboring face's badge, in which case progressively smaller
# tiers are tried so crowded faces still each get a legible badge instead
# of overlapping ones (see VideoWidget._place_badge).
_BADGE_SCALE_TIERS = (1.0, 0.8, 0.65, 0.5)

# The Re-check button is a round icon beside the badge; a full text pill made each
# badge roughly twice as wide, so crowded galleries overlapped and covered faces.
# Its size has a floor so it stays clickable when the badge shrinks.
_RECHECK_DIAMETER = 22
_RECHECK_MIN_DIAMETER = 18
_RECHECK_TOOLTIP = "Re-check this face"


def _tiered_fonts(base_pt: int, min_pt: int = 6) -> dict:
    fonts = {}
    for scale in _BADGE_SCALE_TIERS:
        f = QFont(theme.DISPLAY_FONT, max(int(round(base_pt * scale)), min_pt))
        f.setBold(True)
        fonts[scale] = f
    return fonts


_BADGE_PCT_FONTS = _tiered_fonts(15)
_BADGE_PCT_METRICS = {s: QFontMetrics(f) for s, f in _BADGE_PCT_FONTS.items()}

_BADGE_SUB_FONTS = _tiered_fonts(8)
_BADGE_SUB_METRICS = {s: QFontMetrics(f) for s, f in _BADGE_SUB_FONTS.items()}

_BADGE_FONTS = _tiered_fonts(10)
_BADGE_METRICS = {s: QFontMetrics(f) for s, f in _BADGE_FONTS.items()}

_RECHECK_ICON_FONTS = _tiered_fonts(11, min_pt=8)

_LIVE_FONT = QFont(theme.DISPLAY_FONT, 10)
_LIVE_FONT.setBold(True)
_LIVE_FONT.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)

_IDLE_TITLE_FONT = QFont(theme.DISPLAY_FONT, 18)
_IDLE_TITLE_FONT.setBold(True)
_IDLE_TITLE_FONT.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 6)
_IDLE_HINT_FONT = QFont(theme.DISPLAY_FONT, 10)


def _recheck_diameter(scale: float) -> int:
    return max(int(_RECHECK_DIAMETER * scale), _RECHECK_MIN_DIAMETER)


def _scanline_brush() -> QBrush:
    """Tiled brush darkening every third pixel row, for a CRT scanline overlay."""
    tile = QImage(1, 3, QImage.Format.Format_ARGB32)
    tile.fill(Qt.GlobalColor.transparent)
    tile.setPixelColor(0, 0, QColor(0, 0, 0, 40))
    return QBrush(tile)


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
        self._cached_scaled_pixmap = None
        self._cached_widget_size = None
        self._scanlines = _scanline_brush()
        self._locks = {}  # face_id -> (risk_level, perf_counter time its lock-on started)

        self._idle_scan_y = 0.0
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(_IDLE_TICK_MS)
        self._idle_timer.timeout.connect(self._advance_idle_scan)

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
        # Converted to a pixmap in paintEvent at display size only (~1.4 ms vs ~7 ms for converting
        # the full 2560x1600 frame here), and frames Qt never paints (e.g. hidden in Mini mode) cost nothing.
        self._idle_timer.stop()
        self._current_frame = frame
        self._frame_size = (frame.shape[1], frame.shape[0])
        self._cached_scaled_pixmap = None  # invalidate on new frame
        self._current_results = results
        self._current_pending_bboxes = pending_bboxes
        self._current_rechecking_bboxes = [r.bbox for r in rechecking_results]

        # A lock-on (re)starts when a face gets its first verdict or its verdict changes;
        # faces no longer analyzed drop out, so a finished re-check locks on again.
        now = time.perf_counter()
        locks = {}
        for result in results:
            if result.face_id:
                lock = self._locks.get(result.face_id)
                locks[result.face_id] = lock if lock and lock[0] == result.risk_level else (result.risk_level, now)
        self._locks = locks
        self.update()

    def clear(self):
        """Clear the displayed frame."""
        self._current_frame = None
        self._cached_scaled_pixmap = None
        self._current_results = []
        self._current_pending_bboxes = []
        self._current_rechecking_bboxes = []
        self._recheck_hitboxes = []
        self._locks = {}
        if self.isVisible():
            self._idle_timer.start()
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        if self._current_frame is None:
            self._idle_timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._idle_timer.stop()

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
        pen = QPen(_ZONE_COLOUR, 1, Qt.PenStyle.DashLine)
        fill = QColor(_ZONE_COLOUR)
        fill.setAlpha(25)

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
            theme.draw_corner_brackets(painter, QRectF(wx, wy, ww, wh), _ZONE_COLOUR, 12, 2)
            painter.setFont(_BADGE_SUB_FONTS[1.0])
            painter.setPen(_ZONE_COLOUR)
            painter.drawText(wx + 4, wy - 5, "Detection Zone")

    def paintEvent(self, event):
        painter = QPainter(self)

        if self._current_frame is None:
            self._draw_idle_screen(painter)
            painter.end()
            return

        widget_size = self.size()
        current_w_size = (widget_size.width(), widget_size.height())
        if self._cached_scaled_pixmap is None or self._cached_widget_size != current_w_size:
            self._cached_scaled_pixmap = _fit_pixmap(self._current_frame, *current_w_size)
            self._cached_widget_size = current_w_size
        scaled_pixmap = self._cached_scaled_pixmap

        x_offset = (widget_size.width() - scaled_pixmap.width()) // 2
        y_offset = (widget_size.height() - scaled_pixmap.height()) // 2
        self._display_size = (scaled_pixmap.width(), scaled_pixmap.height())
        self._offset = (x_offset, y_offset)

        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        painter.fillRect(QRect(x_offset, y_offset, *self._display_size), self._scanlines)
        self._draw_scan_band(painter)
        theme.draw_corner_brackets(
            painter, QRectF(x_offset + 1, y_offset + 1, self._display_size[0] - 2, self._display_size[1] - 2),
            _FRAME_BRACKET_COLOUR, 18, 2,
        )
        self._draw_overlays(painter)
        self._draw_selection_zone(painter)
        self._draw_live_indicator(painter)
        painter.end()

    def _draw_scan_band(self, painter: QPainter):
        """Faint cyan band sweeping down the live frame."""
        ox, oy = self._offset
        dw, dh = self._display_size
        centre = oy + (time.perf_counter() % _SCAN_PERIOD_S) / _SCAN_PERIOD_S * dh
        band = QLinearGradient(0, centre - _SCAN_BAND / 2, 0, centre + _SCAN_BAND / 2)
        band.setColorAt(0.0, _SCAN_CLEAR)
        band.setColorAt(0.5, _SCAN_BAND_PEAK)
        band.setColorAt(1.0, _SCAN_CLEAR)
        band_rect = QRectF(ox, centre - _SCAN_BAND / 2, dw, _SCAN_BAND).intersected(QRectF(ox, oy, dw, dh))
        painter.fillRect(band_rect, QBrush(band))

    def _advance_idle_scan(self):
        """Move the idle scan line, repainting only the strip it leaves and the strip it enters."""
        old_y = self._idle_scan_y
        self._idle_scan_y = (time.perf_counter() % _IDLE_PERIOD_S) / _IDLE_PERIOD_S * self.height()
        for y in (old_y, self._idle_scan_y):
            self.update(0, int(y) - _IDLE_TRAIL - 2, self.width(), _IDLE_TRAIL + 4)

    def _draw_idle_screen(self, painter: QPainter):
        """HUD grid with a scan line and a NO SIGNAL banner while no feed is running."""
        painter.setPen(QPen(_GRID_COLOUR, 1))
        for x in range(_GRID_STEP, self.width(), _GRID_STEP):
            painter.drawLine(x, 0, x, self.height())
        for y in range(_GRID_STEP, self.height(), _GRID_STEP):
            painter.drawLine(0, y, self.width(), y)

        y = self._idle_scan_y
        trail = QLinearGradient(0, y - _IDLE_TRAIL, 0, y)
        trail.setColorAt(0.0, _SCAN_CLEAR)
        trail.setColorAt(1.0, _IDLE_TRAIL_PEAK)
        painter.fillRect(QRectF(0, y - _IDLE_TRAIL, self.width(), _IDLE_TRAIL), QBrush(trail))
        painter.setPen(QPen(_IDLE_LINE, 1))
        painter.drawLine(QPointF(0, y), QPointF(self.width(), y))

        centre_y = self.height() // 2
        painter.setFont(_IDLE_TITLE_FONT)
        painter.setPen(QColor(theme.CYAN_DIM))
        painter.drawText(QRect(0, centre_y - 34, self.width(), 34), Qt.AlignmentFlag.AlignCenter, "NO SIGNAL")
        painter.setFont(_IDLE_HINT_FONT)
        painter.setPen(QColor(theme.TEXT_DIM))
        painter.drawText(
            QRect(0, centre_y + 2, self.width(), 24), Qt.AlignmentFlag.AlignCenter,
            "No feed — press Start to begin detection."
        )

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
            box_colour = _RISK_COLOURS.get(result.risk_level, _RISK_COLOURS["REAL"])
            box_rect = QRectF(self._scaled_rect(result.bbox, scale_x, scale_y))
            faint = QColor(box_colour)
            faint.setAlpha(110)
            painter.setPen(QPen(faint, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box_rect)
            bracket = max(6.0, min(box_rect.width(), box_rect.height()) * 0.22)
            lock = self._lock_progress(result.face_id, now)
            spread = (1 - lock) * _LOCK_SPREAD
            bracket_colour = QColor(box_colour)
            bracket_colour.setAlphaF(0.35 + 0.65 * lock)
            theme.draw_corner_brackets(
                painter, box_rect.adjusted(-spread, -spread, spread, spread), bracket_colour, bracket, 3
            )

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

    def _lock_progress(self, face_id: str, now: float) -> float:
        """Eased 0 -> 1 progress of a face's lock-on; 1 once settled or for a face without an id."""
        lock = self._locks.get(face_id)
        if lock is None:
            return 1.0
        p = min(1.0, (now - lock[1]) / _LOCK_S)
        return 1 - (1 - p) ** 3

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
        badge_fg = _RISK_COLOURS.get(result.risk_level, _RISK_COLOURS["REAL"])
        box_rect = self._scaled_rect(result.bbox, scale_x, scale_y)

        p_fake_pct = result.p_fake_raw * 100
        elapsed = now - result.last_updated
        age_str = "just now" if elapsed < 1.0 else f"{elapsed:.0f}s ago"
        pct_text = f"{p_fake_pct:.1f}%"
        sub_text = f"{result.risk_level.upper()}  ·  {age_str}"

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
            return badge_w + int(4 * scale) + _recheck_diameter(scale), h

        combined_rect, scale = self._place_badge(box_rect, measure, placed)
        placed.append(combined_rect)

        badge_w = max(
            _BADGE_PCT_METRICS[scale].horizontalAdvance(pct_text),
            _BADGE_SUB_METRICS[scale].horizontalAdvance(sub_text),
        ) + int(20 * scale)
        rect = QRect(combined_rect.x(), combined_rect.y(), badge_w, combined_rect.height())

        painter.setBrush(QBrush(_BADGE_BG))
        painter.setPen(QPen(badge_fg, 1))
        painter.drawRect(rect)
        painter.fillRect(QRect(rect.x(), rect.y(), max(2, int(3 * scale)), rect.height() + 1), badge_fg)

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
            diameter = _recheck_diameter(scale)
            button_rect = QRect(
                rect.x() + badge_w + int(4 * scale), rect.center().y() - diameter // 2,
                diameter, diameter,
            )
            self._draw_recheck_button(painter, button_rect, scale, result.face_id)

    def _draw_placeholder_badge(self, painter: QPainter, bbox: dict, scale_x: float, scale_y: float,
                                 label: str, box_colour: QColor, text_colour: QColor, placed: list):
        box_rect = self._scaled_rect(bbox, scale_x, scale_y)

        def measure(scale):
            w = _BADGE_METRICS[scale].horizontalAdvance(label) + int(10 * scale)
            h = int(20 * scale)
            return w, h

        rect, scale = self._place_badge(box_rect, measure, placed)
        placed.append(rect)

        painter.setBrush(QBrush(_BADGE_BG))
        painter.setPen(QPen(box_colour, 1))
        painter.drawRect(rect)

        painter.setFont(_BADGE_FONTS[scale])
        painter.setPen(text_colour)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_recheck_button(self, painter: QPainter, rect: QRect, scale: float, face_id: str):
        """Draw the round Re-check icon at an already-resolved rect; records its hitbox for mousePressEvent."""
        painter.setBrush(QBrush(_BADGE_BG))
        painter.setPen(QPen(QColor(theme.CYAN_DIM), 1))
        painter.drawEllipse(rect)

        painter.setFont(_RECHECK_ICON_FONTS[scale])
        painter.setPen(_RECHECK_BUTTON_FG)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "↻")

        self._recheck_hitboxes.append((rect, face_id))

    def event(self, event):
        # The icon has no text, so its tooltip names the action; the widget's own
        # tooltip cannot be used because the buttons are painted, not child widgets.
        if event.type() == QEvent.Type.ToolTip:
            pos = event.pos()
            rect = next((r for r, _ in self._recheck_hitboxes if r.contains(pos)), None)
            if rect is None:
                QToolTip.hideText()
                event.ignore()
            else:
                QToolTip.showText(event.globalPos(), _RECHECK_TOOLTIP, self, rect)
            return True
        return super().event(event)

    def _toggle_blink(self):
        self._blink_on = not self._blink_on
        if self._current_frame is not None:
            self.update()

    def _draw_live_indicator(self, painter: QPainter):
        """Draw blinking LIVE indicator in top-left corner of the video area."""
        circle_x = self._offset[0] + 14
        circle_y = self._offset[1] + 14
        circle_size = 10

        if self._blink_on:
            painter.setBrush(QBrush(QColor(theme.RED)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(circle_x, circle_y, circle_size, circle_size)

        painter.setFont(_LIVE_FONT)
        painter.setPen(QColor(theme.CYAN))
        painter.drawText(
            circle_x + 16, circle_y, 200, circle_size,
            Qt.AlignmentFlag.AlignVCenter,
            "DETECTING"
        )

