"""
gauge_widget.py

Custom-painted semicircular confidence gauge (0-100%), sweeping from
green through yellow/orange to red. Replaces the horizontal progress bar.
"""

import math
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QRectF, QPointF

_TRACK_COLOUR = QColor("#2A2E38")
_SEGMENTS = 48
_STOPS = [
    (0.0, QColor("#22C55E")),   # green  — 0%
    (0.5, QColor("#F59E0B")),   # amber  — 50%
    (1.0, QColor("#EF4444")),   # red    — 100%
]


def _lerp_colour(t: float) -> QColor:
    """Interpolate across the green -> amber -> red stops at position t (0-1)."""
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_STOPS, _STOPS[1:]):
        if t0 <= t <= t1:
            local_t = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = c0.red() + (c1.red() - c0.red()) * local_t
            g = c0.green() + (c1.green() - c0.green()) * local_t
            b = c0.blue() + (c1.blue() - c0.blue()) * local_t
            return QColor(int(r), int(g), int(b))
    return _STOPS[-1][1]


class GaugeWidget(QWidget):
    """Semicircular arc gauge showing a 0-100% confidence score."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._label = "—"  # em dash placeholder before first result
        self.setMinimumHeight(125)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: float, label: str):
        """Update the gauge reading (0-100) and the status label beneath it."""
        value = max(0.0, min(100.0, value))
        if (value, label) == (self._value, self._label):
            return  # called every frame; skip the repaint when the reading is unchanged
        self._value = value
        self._label = label
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin = 14
        # Reserve fixed space below the arc's flat edge for the two text
        # lines (percentage + label) so they can never be clipped by the
        # widget's own bottom edge, regardless of how wide the card is.
        text_area_h = 62
        radius = min((w - 2 * margin) / 2, h - margin - text_area_h)
        radius = max(radius, 30)
        cx = w / 2
        cy = margin + radius
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        pen_width = max(6, int(radius * 0.16))

        # Background track (full range, always visible)
        track_pen = QPen(_TRACK_COLOUR, pen_width)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(rect, 0 * 16, 180 * 16)

        # Coloured spectrum track, drawn as short segments green -> amber -> red
        for i in range(_SEGMENTS):
            t0 = i / _SEGMENTS
            t1 = (i + 1) / _SEGMENTS
            colour = _lerp_colour((t0 + t1) / 2)
            seg_pen = QPen(colour, pen_width)
            seg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(seg_pen)
            start_angle = int((180 - t0 * 180) * 16)
            span_angle = int(-(t1 - t0) * 180 * 16)
            painter.drawArc(rect, start_angle, span_angle)

        # Marker dot at the current value's position
        value_t = self._value / 100.0
        theta = math.radians(180 - value_t * 180)
        mx = cx + radius * math.cos(theta)
        my = cy - radius * math.sin(theta)
        marker_colour = _lerp_colour(value_t)
        painter.setPen(QPen(QColor("#0B0E14"), 3))
        painter.setBrush(marker_colour)
        painter.drawEllipse(QPointF(mx, my), pen_width * 0.55, pen_width * 0.55)

        # 0% / 100% end labels
        painter.setPen(QColor("#6B7280"))
        small_font = QFont("Segoe UI", 8)
        painter.setFont(small_font)
        painter.drawText(
            QRectF(rect.left() - 6, cy + 4, 30, 14),
            Qt.AlignmentFlag.AlignLeft, "0%"
        )
        painter.drawText(
            QRectF(rect.right() - 24, cy + 4, 30, 14),
            Qt.AlignmentFlag.AlignRight, "100%"
        )

        # Centered percentage + label, laid out within the reserved
        # text_area_h band below the arc (see radius calc above) so they
        # can never be clipped by the widget's own bottom edge.
        value_colour = _lerp_colour(value_t) if self._label not in ("—",) else QColor("#6B7280")
        pct_text = f"{self._value:.1f}%" if self._label not in ("—",) else "—"
        pct_font = QFont("Segoe UI", max(14, min(28, int(radius * 0.30))))
        pct_font.setBold(True)
        painter.setFont(pct_font)
        painter.setPen(value_colour)
        pct_rect = QRectF(cx - radius, cy - 20, radius * 2, 36)
        painter.drawText(pct_rect, Qt.AlignmentFlag.AlignCenter, pct_text)

        label_font = QFont("Segoe UI", 10)
        painter.setFont(label_font)
        painter.setPen(QColor("#9CA3AF"))
        label_rect = QRectF(cx - radius, cy + 18, radius * 2, 20)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, self._label)
