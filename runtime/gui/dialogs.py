"""
dialogs.py

Modal dialogs used by the main window, and persistence of user settings.
"""

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
)
from PyQt6.QtCore import Qt, QSettings, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QPixmap, QShortcut

import config

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


def _settings_store() -> QSettings:
    return QSettings("DeepfakeDetector", "DeepfakeDetector")


def load_saved_settings():
    """Apply the values saved by SettingsDialog on top of the config.py defaults."""
    store = _settings_store()
    for key, default in _DEFAULTS.items():
        setattr(config, key, store.value(key, default, type=type(default)))


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
