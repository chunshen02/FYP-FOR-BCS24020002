"""
dialogs.py

Modal dialogs used by the main window, and persistence of user settings.
"""

import html
import os

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSpinBox,
    QCheckBox,
    QStackedWidget,
    QWidget,
)
from PyQt6.QtCore import Qt, QSettings, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QPixmap, QShortcut

import config
from runtime.detection.inference import ENV_PATH, has_api_key

_DIALOG_STYLE = """
    QDialog {
        background-color: #1e1e1e;
    }
    QLabel, QCheckBox {
        color: #dddddd;
    }
    QSpinBox {
        background-color: #2a2a2a;
        color: #dddddd;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 2px 6px;
    }
"""

_SECONDARY_BUTTON_STYLE = """
    QPushButton {
        background-color: transparent;
        border: 1px solid #555555;
        border-radius: 6px;
        color: #aaaaaa;
        font-size: 12px;
        padding: 0 16px;
    }
    QPushButton:hover { background-color: #2a2a2a; }
"""

_PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #E1F5EE;
        border: 1px solid #1D9E75;
        border-radius: 6px;
        color: #085041;
        font-size: 12px;
        font-weight: 500;
        padding: 0 20px;
    }
    QPushButton:hover { background-color: #9FE1CB; }
"""

_SETTING_KEYS = ("FAKE_THRESHOLD", "ALERT_SOUND_ENABLED", "ALERT_NOTIFICATION_ENABLED", "ALERT_AUTO_SCREENSHOT")
# Captured at import, before load_saved_settings() overrides them, so
# "Reset to defaults" restores the values written in config.py.
_DEFAULTS = {key: getattr(config, key) for key in _SETTING_KEYS}

_QUICK_START_SHOWN_KEY = "QUICK_START_SHOWN"


def _settings_store() -> QSettings:
    return QSettings("DeepfakeDetector", "DeepfakeDetector")


def load_saved_settings():
    """Apply the values saved by SettingsDialog on top of the config.py defaults."""
    store = _settings_store()
    for key, default in _DEFAULTS.items():
        setattr(config, key, store.value(key, default, type=type(default)))


def quick_start_pending() -> bool:
    """Whether the Quick Start guide has never been shown on this computer."""
    return not _settings_store().value(_QUICK_START_SHOWN_KEY, False, type=bool)


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #333333;")
    return line


def _dialog_button(text: str, style: str) -> QPushButton:
    button = QPushButton(text)
    button.setFixedHeight(34)
    button.setStyleSheet(style)
    return button


def _guide_text(text: str, style: str = "font-size: 12px; color: #C4C9D4;") -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setWordWrap(True)
    label.setStyleSheet(style)
    return label


def _guide_row(icon: str, text: str) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(10)
    icon_label = QLabel(icon)
    icon_label.setFixedWidth(22)
    icon_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
    icon_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #9CA3AF;")
    row.addWidget(icon_label)
    row.addWidget(_guide_text(text), 1)
    return row


def _legend_row(border: str, name: str, meaning: str) -> QHBoxLayout:
    """A face box drawn in the live overlay's colour and line style, beside what it means."""
    row = QHBoxLayout()
    row.setSpacing(10)
    swatch = QLabel()
    swatch.setFixedSize(22, 16)
    swatch.setStyleSheet(f"border: {border}; border-radius: 3px;")
    row.addWidget(swatch, 0, Qt.AlignmentFlag.AlignTop)
    row.addWidget(_guide_text(f"<b style='color: #E5E7EB;'>{name}</b> — {meaning}"), 1)
    return row


def _guide_page() -> tuple:
    """An empty guide page; returns (page, layout)."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    return page, layout


class SaveSessionDialog(QDialog):
    """Asks the user whether to save or discard the just-finished session."""

    def __init__(self, duration: str, alert_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save session results?")
        self.setModal(True)
        self.setFixedWidth(320)
        self.setStyleSheet(_DIALOG_STYLE)
        # Only the Discard button deletes the session: closing the dialog with
        # the title-bar X or Esc keeps it, so a session with alerts is never lost by accident.
        self.discarded = False
        self._build_ui(duration, alert_count)

    def _discard(self):
        self.discarded = True
        self.reject()

    def _build_ui(self, duration: str, alert_count: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Save session results?")
        title.setStyleSheet("font-size: 14px; font-weight: 500; color: #ffffff;")
        layout.addWidget(title)
        layout.addWidget(_divider())

        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(6)

        def make_row(label: str, value: str) -> QHBoxLayout:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 12px; color: #888888;")
            val = QLabel(value)
            val.setStyleSheet("font-size: 12px; color: #dddddd;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            return row

        stats_layout.addLayout(make_row("Duration", duration))
        stats_layout.addLayout(make_row("Alerts triggered", str(alert_count)))

        result_row = QHBoxLayout()
        result_lbl = QLabel("Result")
        result_lbl.setStyleSheet("font-size: 12px; color: #888888;")
        result_badge = QLabel(
            "Deepfake detected" if alert_count > 0 else "Clean"
        )
        if alert_count > 0:
            result_badge.setStyleSheet(
                """
                font-size: 11px; font-weight: 500;
                background-color: #FCEBEB; color: #A32D2D;
                border-radius: 3px; padding: 2px 8px;
                """
            )
        else:
            result_badge.setStyleSheet(
                """
                font-size: 11px; font-weight: 500;
                background-color: #EAF3DE; color: #3B6D11;
                border-radius: 3px; padding: 2px 8px;
                """
            )
        result_row.addWidget(result_lbl)
        result_row.addStretch()
        result_row.addWidget(result_badge)
        stats_layout.addLayout(result_row)

        layout.addLayout(stats_layout)
        layout.addWidget(_divider())

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        discard_btn = _dialog_button("Discard", _SECONDARY_BUTTON_STYLE)
        discard_btn.clicked.connect(self._discard)

        save_btn = _dialog_button("Save", _PRIMARY_BUTTON_STYLE)
        save_btn.clicked.connect(self.accept)
        save_btn.setDefault(True)

        btn_row.addWidget(discard_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)


class EvidenceDialog(QDialog):
    """Pages through a saved session's alert screenshots, one alert at a time."""

    # Screenshots are full-screen captures; this keeps the dialog on screen
    # while leaving the image large enough to read the faces in it.
    _IMAGE_W, _IMAGE_H = 960, 600

    def __init__(self, started_at: str, shots: list, parent=None):
        """shots: (filename, time, peak P_fake or None, faces that alerted on it) per screenshot, oldest first."""
        super().__init__(parent)
        self.setWindowTitle("Alert screenshots")
        self.setModal(True)
        self.setStyleSheet(_DIALOG_STYLE)
        self._shots = shots
        self._index = 0
        self._build_ui(started_at)
        self._show_shot()

    def _build_ui(self, started_at: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel(f"Alert screenshots — session of {started_at}")
        title.setStyleSheet("font-size: 14px; font-weight: 500; color: #ffffff;")
        layout.addWidget(title)
        layout.addWidget(_divider())

        self._image = QLabel()
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setStyleSheet("background-color: #000000; color: #888888; font-size: 12px;")
        layout.addWidget(self._image)

        self._caption = QLabel()
        self._caption.setStyleSheet("font-size: 12px; color: #aaaaaa;")
        layout.addWidget(self._caption)
        layout.addWidget(_divider())

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._prev_btn = _dialog_button("◀ Previous", _SECONDARY_BUTTON_STYLE)
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn = _dialog_button("Next ▶", _SECONDARY_BUTTON_STYLE)
        self._next_btn.clicked.connect(lambda: self._step(1))
        self._open_btn = _dialog_button("Open file", _SECONDARY_BUTTON_STYLE)
        self._open_btn.clicked.connect(self._open_file)
        close_btn = _dialog_button("Close", _PRIMARY_BUTTON_STYLE)
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        for button in (self._prev_btn, self._next_btn):
            button.setVisible(len(self._shots) > 1)
            btn_row.addWidget(button)
        btn_row.addStretch()
        btn_row.addWidget(self._open_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Window-wide shortcuts, so the arrow keys page even while a button has focus.
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))

    def _path(self) -> str:
        return os.path.join(config.SCREENSHOTS_DIR, self._shots[self._index][0])

    def _show_shot(self):
        if not self._shots:
            self._image.setFixedSize(self._IMAGE_W, 120)
            self._image.setText("No alert screenshots were saved for this session.")
            self._caption.setVisible(False)
            self._open_btn.setVisible(False)
            return

        filename, shot_time, peak, new_faces = self._shots[self._index]
        self._image.setFixedSize(self._IMAGE_W, self._IMAGE_H)
        pixmap = QPixmap(self._path())
        if pixmap.isNull():
            self._image.setPixmap(QPixmap())
            self._image.setText(f"Screenshot file not found: {filename}")
        else:
            self._image.setPixmap(pixmap.scaled(
                self._IMAGE_W, self._IMAGE_H,
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            ))
        self._open_btn.setEnabled(not pixmap.isNull())

        caption = f"Screenshot {self._index + 1} of {len(self._shots)}  ·  {shot_time}"
        if new_faces > 1:
            caption += f"  ·  {new_faces} new deepfake faces"
        if peak is not None:
            caption += f"  ·  Peak confidence {peak:.1%}"
        self._caption.setText(caption)
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < len(self._shots) - 1)

    def _step(self, delta: int):
        self._index = max(0, min(self._index + delta, len(self._shots) - 1))
        self._show_shot()

    def _open_file(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._path()))


class SettingsDialog(QDialog):
    """Edits the deepfake threshold and alert cues; Save applies them immediately and persists them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedWidth(360)
        self.setStyleSheet(_DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 14px; font-weight: 500; color: #ffffff;")
        layout.addWidget(title)
        layout.addWidget(_divider())

        threshold_row = QHBoxLayout()
        threshold_label = QLabel("Deepfake threshold")
        threshold_label.setStyleSheet("font-size: 12px;")
        threshold_row.addWidget(threshold_label)
        threshold_row.addStretch()
        self._threshold = QSpinBox()
        self._threshold.setRange(5, 95)
        self._threshold.setSingleStep(5)
        self._threshold.setSuffix(" %")
        threshold_row.addWidget(self._threshold)
        layout.addLayout(threshold_row)

        hint = QLabel(
            "Faces scoring at or above this are flagged DEEPFAKE. Applies to new "
            "verdicts; use the ↻ button beside a face's verdict to re-classify it."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: #888888;")
        layout.addWidget(hint)

        self._sound = QCheckBox("Beep on alert")
        self._notification = QCheckBox("Desktop notification on alert")
        self._screenshot = QCheckBox("Save evidence screenshot on alert")
        for checkbox in (self._sound, self._notification, self._screenshot):
            checkbox.setStyleSheet("font-size: 12px;")
            layout.addWidget(checkbox)

        layout.addWidget(_divider())

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        reset_btn = _dialog_button("Reset to defaults", _SECONDARY_BUTTON_STYLE)
        reset_btn.clicked.connect(lambda: self._show_values(_DEFAULTS))
        cancel_btn = _dialog_button("Cancel", _SECONDARY_BUTTON_STYLE)
        cancel_btn.clicked.connect(self.reject)
        save_btn = _dialog_button("Save", _PRIMARY_BUTTON_STYLE)
        save_btn.clicked.connect(self._on_save)
        save_btn.setDefault(True)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self._show_values({key: getattr(config, key) for key in _SETTING_KEYS})

    def _show_values(self, values: dict):
        self._threshold.setValue(round(values["FAKE_THRESHOLD"] * 100))
        self._sound.setChecked(values["ALERT_SOUND_ENABLED"])
        self._notification.setChecked(values["ALERT_NOTIFICATION_ENABLED"])
        self._screenshot.setChecked(values["ALERT_AUTO_SCREENSHOT"])

    def _on_save(self):
        values = {
            "FAKE_THRESHOLD": self._threshold.value() / 100,
            "ALERT_SOUND_ENABLED": self._sound.isChecked(),
            "ALERT_NOTIFICATION_ENABLED": self._notification.isChecked(),
            "ALERT_AUTO_SCREENSHOT": self._screenshot.isChecked(),
        }
        store = _settings_store()
        for key, value in values.items():
            setattr(config, key, value)
            store.setValue(key, value)
        self.accept()


class QuickStartDialog(QDialog):
    """Pages a first-time user through setup, running a session, reading the results and the extra tools."""

    _WIDTH, _MARGIN = 480, 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick start guide")
        self.setModal(True)
        self.setFixedWidth(self._WIDTH)
        self.setStyleSheet(_DIALOG_STYLE)
        # Marked as shown on opening, so a guide skipped or closed early is not forced on every launch.
        _settings_store().setValue(_QUICK_START_SHOWN_KEY, True)

        pages = (
            ("Welcome to Deepfake Detector", self._welcome_page()),
            ("Start a session", self._session_page()),
            ("Reading the results", self._results_page()),
            ("Handy tools", self._tools_page()),
        )
        self._titles = [title for title, _ in pages]
        self._pages = QStackedWidget()
        for _, page in pages:
            self._pages.addWidget(page)
        # A stacked widget sizes to its pages' size hints, which assume word-wrapped text is narrow and
        # leave a large gap below it; fit to the tallest page's wrapped height at the real width instead.
        page_width = self._WIDTH - 2 * self._MARGIN
        self._pages.setFixedHeight(max(page.heightForWidth(page_width) for _, page in pages))
        self._build_ui()
        self._show_page(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(self._MARGIN, self._MARGIN, self._MARGIN, self._MARGIN)

        self._step_label = QLabel()
        self._step_label.setStyleSheet("font-size: 10px; font-weight: 600; letter-spacing: 1px; color: #6B7280;")
        layout.addWidget(self._step_label)
        self._title = QLabel()
        self._title.setStyleSheet("font-size: 14px; font-weight: 500; color: #ffffff;")
        layout.addWidget(self._title)
        layout.addWidget(_divider())
        layout.addWidget(self._pages)
        layout.addWidget(_divider())

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._dots = QLabel()
        self._dots.setStyleSheet("font-size: 9px;")
        btn_row.addWidget(self._dots)
        btn_row.addStretch()
        self._skip_btn = _dialog_button("Skip", _SECONDARY_BUTTON_STYLE)
        self._skip_btn.clicked.connect(self.reject)
        self._back_btn = _dialog_button("◀ Back", _SECONDARY_BUTTON_STYLE)
        self._back_btn.clicked.connect(lambda: self._show_page(self._pages.currentIndex() - 1))
        self._next_btn = _dialog_button("Next ▶", _PRIMARY_BUTTON_STYLE)
        self._next_btn.clicked.connect(self._on_next)
        self._next_btn.setDefault(True)
        for button in (self._skip_btn, self._back_btn, self._next_btn):
            btn_row.addWidget(button)
        layout.addLayout(btn_row)

    def _show_page(self, index: int):
        count = self._pages.count()
        last = index == count - 1
        self._pages.setCurrentIndex(index)
        self._step_label.setText(f"STEP {index + 1} OF {count}")
        self._title.setText(self._titles[index])
        self._dots.setText("&nbsp;&nbsp;".join(
            f"<span style='color: {'#22C55E' if i == index else '#4B5563'};'>●</span>" for i in range(count)
        ))
        self._back_btn.setVisible(index > 0)
        self._skip_btn.setVisible(not last)
        self._next_btn.setText("Get started" if last else "Next ▶")

    def _on_next(self):
        index = self._pages.currentIndex()
        if index == self._pages.count() - 1:
            self.accept()
        else:
            self._show_page(index + 1)

    @staticmethod
    def _welcome_page() -> QWidget:
        page, layout = _guide_page()
        layout.addWidget(_guide_text(
            "This app watches your screen during a video call and warns you when a face on it "
            "looks like a deepfake.",
            "font-size: 12px; color: #E5E7EB;",
        ))
        layout.addLayout(_guide_row(
            "🖥", f"It captures your <b>primary screen</b> about {config.DISPLAY_FPS} times a second "
                  "and finds every face on it."
        ))
        layout.addLayout(_guide_row(
            "☁", "Each face is checked by the <b>BitMind</b> deepfake detection API, so an internet "
                 "connection is needed. Only the cropped face image is sent."
        ))
        layout.addLayout(_guide_row("🔒", "Session history and alert screenshots stay on this computer."))

        if has_api_key():
            text = "✓&nbsp; BitMind API key found — you are ready to start."
            background, border, colour = "#0E1F15", "#1D5C35", "#4ADE80"
        else:
            text = (f"✗&nbsp; No BitMind API key found. Add the line <b>BITMIND_API_KEY=your_key</b> "
                    f"to <b>{html.escape(ENV_PATH)}</b>, then restart the app.")
            background, border, colour = "#2A1414", "#5C2020", "#FCA5A5"
        key_check = QFrame()
        key_check.setObjectName("key_check")
        key_check.setStyleSheet(
            f"QFrame#key_check {{ background-color: {background}; border: 1px solid {border}; border-radius: 6px; }}"
        )
        key_layout = QVBoxLayout(key_check)
        key_layout.setContentsMargins(10, 8, 10, 8)
        key_label = _guide_text(text, f"font-size: 11px; color: {colour};")
        key_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)   # so the path can be copied
        key_layout.addWidget(key_label)
        layout.addWidget(key_check)
        layout.addStretch()
        return page

    @staticmethod
    def _session_page() -> QWidget:
        page, layout = _guide_page()
        layout.addLayout(_guide_row(
            "1", "Open your video call and put its window on your <b>primary screen</b>. Gallery and "
                 "speaker views both work; bigger faces give more reliable results."
        ))
        layout.addLayout(_guide_row(
            "2", "Press <b>▶ Start</b>. The status turns <b>Active</b> and the Live Preview shows what the "
                 "app sees. The app's own window is hidden from the capture, so it never checks itself."
        ))
        layout.addLayout(_guide_row(
            "3", "Press <b>⛶ Mini Mode</b> to shrink the app to a small always-on-top panel with live face "
                 "thumbnails, so the call stays in view."
        ))
        layout.addLayout(_guide_row(
            "4", "Press <b>◻ Stop</b> when the call ends, then <b>Save</b> the session to keep it in "
                 "Session History, or <b>Discard</b> it."
        ))
        layout.addStretch()
        return page

    @staticmethod
    def _results_page() -> QWidget:
        page, layout = _guide_page()
        threshold = f"{config.FAKE_THRESHOLD * 100:.0f}%"
        layout.addLayout(_legend_row("2px dashed #6B7280", "Analyzing…", "face found, verdict not ready yet."))
        layout.addLayout(_legend_row("2px solid #22C55E", "REAL", f"deepfake score below {threshold}."))
        layout.addLayout(_legend_row("2px solid #EF4444", "DEEPFAKE", f"deepfake score of {threshold} or more."))
        layout.addLayout(_legend_row("2px dashed #F59E0B", "Re-checking…", "a fresh verdict was requested."))
        layout.addWidget(_divider())
        layout.addLayout(_guide_row(
            "%", "The percentage on each badge is BitMind's deepfake score. A suspicious face is checked "
                 f"up to {config.CONFIRM_VERDICTS} times before it is called DEEPFAKE, so it can stay on "
                 "Analyzing… for 10–15 seconds."
        ))
        layout.addLayout(_guide_row(
            "⚠", "When a face stays DEEPFAKE, an <b>alert</b> appears with a beep, a desktop notification and "
                 "a flashing taskbar icon, and an evidence screenshot is saved. Press <b>Acknowledge</b> to "
                 "dismiss it."
        ))
        layout.addLayout(_guide_row(
            "ⓘ", "An alert is a warning, not proof. Confirm who you are talking to in another way before "
                 "acting on it."
        ))
        layout.addStretch()
        return page

    @staticmethod
    def _tools_page() -> QWidget:
        page, layout = _guide_page()
        layout.addLayout(_guide_row(
            "🔲", "<b>Detection zone</b> — while detecting, drag a rectangle on the Live Preview to check only "
                 "that area, such as the call window. Right-click or press <b>✕ Clear Zone</b> to remove it."
        ))
        layout.addLayout(_guide_row(
            "↻", "<b>Re-check</b> — a face keeps its verdict while it stays on screen. Click ↻ beside its "
                 "badge to get a fresh one."
        ))
        layout.addLayout(_guide_row(
            "⚙", "<b>Settings</b> — change the deepfake threshold, and turn the alert beep, notification and "
                 "evidence screenshot on or off."
        ))
        layout.addLayout(_guide_row(
            "🗂", "<b>Session History</b> — double-click a saved session to see its alert screenshots, or "
                  "select it and press <b>Export CSV</b> to save its event log."
        ))
        layout.addLayout(_guide_row("?", "Open this guide again at any time with the <b>?</b> button."))
        layout.addStretch()
        return page
