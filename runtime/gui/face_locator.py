"""
face_locator.py

Mini Mode's on-screen face highlight: a borderless, see-through window
drawn over a face's real position on the screen, showing its verdict box
and score for a few seconds before fading out.
"""

from PyQt6.QtCore import QAbstractAnimation, QPropertyAnimation, QRect, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QWidget

from runtime.capture.window_affinity import WDA_EXCLUDEFROMCAPTURE, apply_capture_exclusion
from runtime.gui import theme
from runtime.gui.face_thumbnail import face_details_html

_RISK_COLOURS = {
    "REAL": QColor(theme.GREEN),
    "DEEPFAKE": QColor(theme.RED),
}
_BADGE_BG = QColor(theme.SURFACE)
_BADGE_BG.setAlpha(225)

_HOLD_MS = 3000         # fully visible this long after a click
_LEAVE_HOLD_MS = 1500   # after the pointer leaves, before the fade starts again
_FADE_MS = 700
_MARGIN = 8             # px between the face and its corner brackets
_BADGE_GAP = 4
_BADGE_H = 40
_HINT = "Click to close"

_PCT_FONT = QFont(theme.DISPLAY_FONT, 14)
_PCT_FONT.setBold(True)
_SUB_FONT = QFont(theme.DISPLAY_FONT, 8)
_SUB_FONT.setBold(True)


class FaceLocator(QWidget):
    """Highlights one face at its real screen position, then fades out; hovering pauses the fade, a click closes it."""

    def __init__(self, parent):
        super().__init__(
            parent,
            Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)   # keep focus on the call / video
        self._capture_excluded = None   # decided on first use, once the native window exists
        self._result = None
        self._face_rect = QRect()       # in this window's coordinates
        self._badge_rect = QRect()

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(_FADE_MS)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.hide)
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._fade.start)

    def show_face(self, result) -> bool:
        """Highlight result's face; False when this window cannot be kept out of screen capture."""
        if self._capture_excluded is None:
            # Unless the compositor leaves this window out of every capture, the box drawn over the face
            # would reach the detector's frames (or, under WDA_MONITOR, black the face out of them).
            _, affinity = apply_capture_exclusion(int(self.winId()))
            self._capture_excluded = affinity == WDA_EXCLUDEFROMCAPTURE
        if not self._capture_excluded:
            return False
        self._place(result)
        self._hold(_HOLD_MS)
        self.show()
        self.raise_()
        return True

    def follow(self, verdicts: list):
        """Keep the highlight on its face as it moves, and fade out at once when the face leaves the screen."""
        if not self.isVisible():
            return
        result = next((r for r in verdicts if r.face_id == self._result.face_id), None)
        if result is None:
            if self._fade.state() != QAbstractAnimation.State.Running:
                self._hold_timer.stop()
                self._fade.start()
        elif (result.bbox, result.last_updated) != (self._result.bbox, self._result.last_updated):
            self._place(result)

    def dismiss(self):
        """Hide immediately."""
        self._hold_timer.stop()
        self._fade.stop()
        self.hide()

    def _hold(self, ms: int):
        """Show at full opacity, then start fading after ms."""
        self._fade.stop()
        self.setWindowOpacity(1.0)
        self._hold_timer.start(ms)

    def _place(self, result):
        """Fit the window around result's face and the score badge beside it."""
        self._result = result
        screen = QApplication.primaryScreen()
        geo = screen.geometry()
        ratio = screen.devicePixelRatio()
        # Frames are grabbed from the primary monitor in physical pixels; Qt positions windows in logical ones.
        bbox = result.bbox
        face = QRect(
            geo.x() + round(bbox["x"] / ratio), geo.y() + round(bbox["y"] / ratio),
            round(bbox["w"] / ratio), round(bbox["h"] / ratio),
        )
        frame = face.adjusted(-_MARGIN, -_MARGIN, _MARGIN, _MARGIN)

        pct_text, sub_text = self._badge_text(result)
        badge_w = max(
            QFontMetrics(_PCT_FONT).horizontalAdvance(pct_text), QFontMetrics(_SUB_FONT).horizontalAdvance(sub_text)
        ) + 20
        badge_y = frame.bottom() + 1 + _BADGE_GAP
        if badge_y + _BADGE_H > geo.bottom():
            badge_y = frame.top() - _BADGE_GAP - _BADGE_H
        badge = QRect(frame.left(), badge_y, badge_w, _BADGE_H)

        window = frame.united(badge)
        self.setGeometry(window)
        self._face_rect = frame.translated(-window.topLeft())
        self._badge_rect = badge.translated(-window.topLeft())
        self.setToolTip(face_details_html(result, _HINT))
        self.update()

    @staticmethod
    def _badge_text(result) -> tuple:
        return f"{result.p_fake_raw * 100:.1f}%", result.risk_level

    def enterEvent(self, event):
        self._hold_timer.stop()
        self._fade.stop()
        self.setWindowOpacity(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.isVisible():   # hiding on a click also sends a leave event
            self._hold(_LEAVE_HOLD_MS)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.dismiss()

    def paintEvent(self, event):
        if self._result is None:
            return
        painter = QPainter(self)
        colour = _RISK_COLOURS.get(self._result.risk_level, QColor(theme.GREY))

        # The pointer passes through fully transparent pixels of a see-through window, so this
        # faint tint is what lets hovering anywhere on the face show its tooltip.
        tint = QColor(colour)
        tint.setAlpha(30)
        painter.fillRect(self._face_rect, tint)
        bracket = max(8.0, min(self._face_rect.width(), self._face_rect.height()) * 0.22)
        theme.draw_corner_brackets(painter, QRectF(self._face_rect).adjusted(1.5, 1.5, -1.5, -1.5), colour, bracket, 3)

        badge = self._badge_rect.adjusted(0, 0, -1, -1)
        painter.setBrush(QBrush(_BADGE_BG))
        painter.setPen(QPen(colour, 1))
        painter.drawRect(badge)
        painter.fillRect(QRect(badge.x(), badge.y(), 3, badge.height() + 1), colour)

        pct_text, sub_text = self._badge_text(self._result)
        painter.setFont(_PCT_FONT)
        painter.drawText(QRect(badge.x(), badge.y() + 2, badge.width(), 22), Qt.AlignmentFlag.AlignCenter, pct_text)
        painter.setFont(_SUB_FONT)
        painter.drawText(QRect(badge.x(), badge.y() + 23, badge.width(), 14), Qt.AlignmentFlag.AlignCenter, sub_text)
        painter.end()
