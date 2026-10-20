"""
gauge_widget.py

Custom-painted semicircular confidence gauge (0-100%): a segmented HUD arc
that sweeps from green through amber to red as the reading changes, with the
deepfake threshold marked on it. Replaces the horizontal progress bar.
"""

import math
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPixmap
from PyQt6.QtCore import Qt, QRectF, QPointF, QEasingCurve, QVariantAnimation

import config
from runtime.gui import theme

_UNLIT_COLOUR = QColor(theme.BORDER)
_TICK_COLOUR = QColor(theme.CYAN_DIM)
_THRESHOLD_COLOUR = QColor(theme.MAGENTA)
_SEGMENTS = 30
_SEGMENT_GAP_DEG = 1.6
_STOPS = [
    (0.0, QColor(theme.GREEN)),   # 0%
    (0.5, QColor(theme.AMBER)),   # 50%
    (1.0, QColor(theme.RED)),     # 100%
]

_SWEEP_MS = 450
_FLASH_MS = 600

_END_LABEL_FONT = QFont(theme.MONO_FONT, 8)
_LABEL_FONT = QFont(theme.DISPLAY_FONT, 9)
_LABEL_FONT.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)


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
    return QColor(_STOPS[-1][1])


def _arc_point(cx: float, cy: float, r: float, t: float) -> QPointF:
    """Point at radius r and gauge position t (0 = left end, 1 = right end)."""
    theta = math.radians(180 - t * 180)
    return QPointF(cx + r * math.cos(theta), cy - r * math.sin(theta))


def _segment_angles(i: int) -> tuple:
    """drawArc start and span (1/16 degree) of segment i, leaving a gap before the next one."""
    step = 180 / _SEGMENTS
    return int((180 - i * step - _SEGMENT_GAP_DEG / 2) * 16), int(-(step - _SEGMENT_GAP_DEG) * 16)


class GaugeWidget(QWidget):
    """Semicircular arc gauge showing a 0-100% confidence score."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0   # latest reading
        self._shown = 0.0   # reading currently drawn, swept towards _value
        self._flash = 0.0   # 1 -> 0 glow boost after the verdict changes
        self._label = "—"  # em dash placeholder before first result
        self._static_key = None
        self._static_pixmap = None
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._sweep = QVariantAnimation(self)
        self._sweep.setDuration(_SWEEP_MS)
        self._sweep.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._sweep.valueChanged.connect(self._on_sweep)

        self._flash_anim = QVariantAnimation(self)
        self._flash_anim.setStartValue(1.0)
        self._flash_anim.setEndValue(0.0)
        self._flash_anim.setDuration(_FLASH_MS)
        self._flash_anim.valueChanged.connect(self._on_flash)

    def set_value(self, value: float, label: str):
        """Update the gauge reading (0-100) and the status label beneath it; the arc and number sweep to it."""
        value = float(max(0.0, min(100.0, value)))
        if (value, label) == (self._value, self._label):
            return  # called every frame; skip the repaint when the reading is unchanged
        verdict_changed = label != self._label and label != "—"
        self._value = value
        self._label = label
        self._sweep.stop()

        if not self.isVisible():
            # Hidden in Mini mode: nothing would be seen, so land on the reading without animating.
            self._shown = value
            self.update()
            return

        if verdict_changed:
            self._flash_anim.stop()
            self._flash_anim.start()
        if self._shown == value:
            self.update()
        else:
            self._sweep.setStartValue(self._shown)
            self._sweep.setEndValue(value)
            self._sweep.start()

    def _on_sweep(self, shown: float):
        self._shown = shown
        self.update()

    def _on_flash(self, flash: float):
        self._flash = flash
        self.update()

    def _geometry(self) -> tuple:
        """(cx, cy, radius, arc rect, pen width) for the current widget size."""
        w, h = self.width(), self.height()
        margin = 14
        # Reserve fixed space below the arc's flat edge for the two text
        # lines (percentage + label) so they can never be clipped by the
        # widget's own bottom edge, regardless of how wide the card is.
        text_area_h = 62
        radius = max(min((w - 2 * margin) / 2, h - margin - text_area_h), 30)
        cx = w / 2
        cy = margin + radius
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        return cx, cy, radius, rect, max(6, int(radius * 0.16))

    def _static_layer(self) -> QPixmap:
        """Unlit segments, scale ticks and end labels, redrawn only when the widget's size changes."""
        dpr = self.devicePixelRatioF()
        key = (self.width(), self.height(), dpr)
        if key == self._static_key:
            return self._static_pixmap

        pixmap = QPixmap(math.ceil(self.width() * dpr), math.ceil(self.height() * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, radius, rect, pen_width = self._geometry()

        painter.setPen(QPen(_UNLIT_COLOUR, pen_width, cap=Qt.PenCapStyle.FlatCap))
        for i in range(_SEGMENTS):
            painter.drawArc(rect, *_segment_angles(i))

        # Scale ticks every 10% inside the arc, longer at 0 / 50 / 100%
        tick_r = radius - pen_width / 2 - 5
        painter.setPen(QPen(_TICK_COLOUR, 1))
        for k in range(11):
            length = 5 if k % 5 == 0 else 3
            painter.drawLine(_arc_point(cx, cy, tick_r, k / 10), _arc_point(cx, cy, tick_r - length, k / 10))

        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.setFont(_END_LABEL_FONT)
        painter.drawText(QRectF(rect.left() - 6, cy + 4, 30, 14), Qt.AlignmentFlag.AlignLeft, "0%")
        painter.drawText(QRectF(rect.right() - 24, cy + 4, 30, 14), Qt.AlignmentFlag.AlignRight, "100%")
        painter.end()

        self._static_key = key
        self._static_pixmap = pixmap
        return pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._static_layer())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, radius, rect, pen_width = self._geometry()
        shown_t = self._shown / 100.0

        # Lit segments go over the cached unlit ones; the leading segment fades in as the sweep
        # reaches it, so the arc moves smoothly instead of jumping a whole segment at a time.
        lit = shown_t * _SEGMENTS
        for i in range(math.ceil(lit)):
            fill = min(1.0, lit - i)
            colour = _lerp_colour((i + 0.5) / _SEGMENTS)
            glow = QColor(colour)
            glow.setAlphaF(fill * (0.22 + 0.5 * self._flash))
            colour.setAlphaF(fill)
            start_angle, span_angle = _segment_angles(i)
            painter.setPen(QPen(glow, pen_width + 6 + 4 * self._flash, cap=Qt.PenCapStyle.FlatCap))
            painter.drawArc(rect, start_angle, span_angle)
            painter.setPen(QPen(colour, pen_width, cap=Qt.PenCapStyle.FlatCap))
            painter.drawArc(rect, start_angle, span_angle)

        # Deepfake threshold: a reading past this line is called DEEPFAKE. Drawn over the lit
        # segments so it stays visible, and read live so a new threshold from Settings shows at once.
        painter.setPen(QPen(_THRESHOLD_COLOUR, 2))
        painter.drawLine(
            _arc_point(cx, cy, radius - pen_width / 2 - 4, config.FAKE_THRESHOLD),
            _arc_point(cx, cy, radius + pen_width / 2 + 4, config.FAKE_THRESHOLD),
        )

        # Centered percentage + label, laid out within the reserved band
        # below the arc (see _geometry) so they can never be clipped.
        has_value = self._label != "—"
        value_colour = _lerp_colour(shown_t) if has_value else QColor(theme.TEXT_MUTED)
        pct_text = f"{self._shown:.1f}%" if has_value else "—"
        pct_font = QFont(theme.DISPLAY_FONT, max(14, min(28, int(radius * 0.30))))
        pct_font.setBold(True)
        painter.setFont(pct_font)
        pct_rect = QRectF(cx - radius, cy - 20, radius * 2, 36)
        if has_value:
            glow = QColor(value_colour)
            glow.setAlpha(int(70 + 150 * self._flash))
            painter.setPen(glow)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                painter.drawText(pct_rect.translated(dx, dy), Qt.AlignmentFlag.AlignCenter, pct_text)
        painter.setPen(value_colour)
        painter.drawText(pct_rect, Qt.AlignmentFlag.AlignCenter, pct_text)

        painter.setFont(_LABEL_FONT)
        painter.setPen(value_colour if has_value else QColor(theme.TEXT_DIM))
        label_text = f"[ {self._label.upper()} ]" if has_value else self._label
        label_rect = QRectF(cx - radius, cy + 18, radius * 2, 20)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label_text)
