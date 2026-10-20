"""
theme.py

Cyberpunk HUD palette, fonts and shared painting helpers used by every GUI module.
"""

from PyQt6.QtCore import QPointF, QRectF, Qt, QVariantAnimation
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

BG = "#05070D"
SURFACE = "#0A0F1A"
BORDER = "#16303D"
GRID = "#0C1A24"

CYAN = "#00F0FF"
CYAN_DIM = "#0B5563"
MAGENTA = "#FF2BD6"

# Verdict hues stay green / red / amber / grey so the user guide's colour names remain true.
GREEN = "#00FF9C"
RED = "#FF2A55"
AMBER = "#FFB000"
GREY = "#5C6F82"
AMBER_FILL = "#161002"

TEXT = "#D8F6FF"
TEXT_DIM = "#7A97A8"
TEXT_MUTED = "#3F5868"

DISPLAY_FONT = "Bahnschrift"
MONO_FONT = "Consolas"

SECTION_TITLE_STYLE = (
    f"color: {CYAN}; font-family: '{DISPLAY_FONT}'; font-size: 10px; font-weight: 600; letter-spacing: 2px;"
)


def rgba(colour: str, alpha: float) -> str:
    """QSS rgba() of a hex colour at the given 0-1 opacity."""
    c = QColor(colour)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {int(alpha * 255)})"


def badge_style(colour: str, font_px: int = 10) -> str:
    """Neon outlined badge: tinted fill, coloured border and text."""
    return (
        f"background-color: {rgba(colour, 0.12)}; color: {colour}; border: 1px solid {colour}; "
        f"padding: 1px 6px; font-size: {font_px}px; font-weight: 600; letter-spacing: 1px;"
    )


def chamfered_path(rect: QRectF, cut: float) -> QPainterPath:
    """Rectangle with its top-left and bottom-right corners cut at 45 degrees."""
    path = QPainterPath()
    path.moveTo(rect.left() + cut, rect.top())
    path.lineTo(rect.right(), rect.top())
    path.lineTo(rect.right(), rect.bottom() - cut)
    path.lineTo(rect.right() - cut, rect.bottom())
    path.lineTo(rect.left(), rect.bottom())
    path.lineTo(rect.left(), rect.top() + cut)
    path.closeSubpath()
    return path


def draw_corner_brackets(painter: QPainter, rect: QRectF, colour: QColor, length: float, width: float):
    """Draw HUD-style L brackets on the four corners of rect."""
    painter.setPen(QPen(colour, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
    for x, y, dx, dy in ((left, top, 1, 1), (right, top, -1, 1), (left, bottom, 1, -1), (right, bottom, -1, -1)):
        painter.drawLine(QPointF(x, y), QPointF(x + dx * length, y))
        painter.drawLine(QPointF(x, y), QPointF(x, y + dy * length))


class HudCard(QWidget):
    """Panel with chamfered corners, a faint border and bright accent strokes on the cut corners."""

    _CUT = 10
    _ACCENT_RUN = 26
    _PULSE_MS = 600
    _PULSES = 3

    def __init__(self, accent: str = CYAN, fill: str = SURFACE, parent=None):
        super().__init__(parent)
        self._accent = QColor(accent)
        self._fill = QColor(fill)
        self._border = QColor(accent)
        self._border.setAlpha(60)

        self._glow = 0.0
        self._pulse = QVariantAnimation(self)
        self._pulse.setStartValue(0.0)
        self._pulse.setKeyValueAt(0.5, 1.0)
        self._pulse.setEndValue(0.0)
        self._pulse.setDuration(self._PULSE_MS)
        self._pulse.setLoopCount(self._PULSES)
        self._pulse.valueChanged.connect(self._on_pulse)

    def pulse(self):
        """Flash the border glow a few times, then settle, to draw the eye to this panel."""
        self._pulse.stop()
        self._pulse.start()

    def _on_pulse(self, glow: float):
        self._glow = glow
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = chamfered_path(rect, self._CUT)
        painter.fillPath(path, self._fill)
        if self._glow > 0:
            tint = QColor(self._accent)
            tint.setAlphaF(0.1 * self._glow)
            painter.fillPath(path, tint)
            # The widget clips the outer half of this wide stroke, leaving a glow inside the border.
            glow = QColor(self._accent)
            glow.setAlphaF(0.6 * self._glow)
            painter.setPen(QPen(glow, 1 + 5 * self._glow))
            painter.drawPath(path)
        painter.setPen(QPen(self._border, 1))
        painter.drawPath(path)

        cut, run = self._CUT, self._ACCENT_RUN
        painter.setPen(QPen(self._accent, 2))
        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        painter.drawPolyline([
            QPointF(left, top + cut + run / 2), QPointF(left, top + cut),
            QPointF(left + cut, top), QPointF(left + cut + run, top),
        ])
        painter.drawPolyline([
            QPointF(right, bottom - cut - run / 2), QPointF(right, bottom - cut),
            QPointF(right - cut, bottom), QPointF(right - cut - run, bottom),
        ])
        painter.end()
