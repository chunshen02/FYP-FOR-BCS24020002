"""
dialogs.py

Modal dialogs used by the main window.
"""

import os
import random
import sys

import cv2
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
)
from PyQt6.QtCore import Qt, QTimer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import BitmindAPIError
from runtime.gui.video_widget import VideoWidget


class SaveSessionDialog(QDialog):
    """Asks the user whether to save or discard the just-finished session."""

    def __init__(self, duration: str, alert_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save session results?")
        self.setModal(True)
        self.setFixedWidth(320)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #dddddd;
            }
            """
        )
        self._build_ui(duration, alert_count)

    def _build_ui(self, duration: str, alert_count: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Save session results?")
        title.setStyleSheet("font-size: 14px; font-weight: 500; color: #ffffff;")
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #333333;")
        layout.addWidget(line)

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

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("color: #333333;")
        layout.addWidget(line2)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        discard_btn = QPushButton("Discard")
        discard_btn.setFixedHeight(34)
        discard_btn.setStyleSheet(
            """
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
        )
        discard_btn.clicked.connect(self.reject)

        save_btn = QPushButton("Save")
        save_btn.setFixedHeight(34)
        save_btn.setStyleSheet(
            """
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
        )
        save_btn.clicked.connect(self.accept)
        save_btn.setDefault(True)

        btn_row.addWidget(discard_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)


_PRACTICE_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _list_practice_clips() -> list:
    """Return [(path, label), ...] for all video clips under config.PRACTICE_CLIPS_DIR."""
    clips = []
    for label, subdir in (("REAL", "real"), ("DEEPFAKE", "deepfake")):
        folder = os.path.join(config.PRACTICE_CLIPS_DIR, subdir)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if os.path.splitext(f)[1].lower() in _PRACTICE_VIDEO_EXTS:
                clips.append((os.path.join(folder, f), label))
    return clips


def _sample_clip_frames(path: str, n: int) -> list:
    """Return up to n evenly-spaced BGR frames from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []
    step = max(1, total // n)
    frames = []
    for i in range(n):
        pos = min(i * step, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _classify_clip(path: str, detector, preprocessor: Preprocessor, engine) -> tuple:
    """
    Classify a clip via its worst-scoring sampled frame/face, reusing the
    already-initialized shared detector/engine. A BitmindAPIError on one
    frame/face is skipped rather than aborting the whole reveal.

    Returns:
        (risk_level, p_fake_raw), or (None, None) if nothing could be classified.
    """
    best = None
    for frame in _sample_clip_frames(path, config.EVAL_FRAMES_PER_VIDEO):
        h, w = frame.shape[:2]
        scale = config.DETECTION_UPSCALE
        scaled = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        face_crops = detector.detect(scaled)
        crops = [crop for crop, _ in face_crops] if face_crops else [frame]
        for crop in crops:
            try:
                jpeg = preprocessor.process(crop)
                result = engine.run_single(jpeg, bbox={}, face_id="practice")
            except BitmindAPIError:
                continue
            if best is None or result.p_fake_raw > best.p_fake_raw:
                best = result
    if best is None:
        return None, None
    return best.risk_level, best.p_fake_raw


class PracticeModeDialog(QDialog):
    """
    Pre-test/post-test practice mode for user studies: shows a pre-loaded
    clip, the participant guesses Real/Fake without the tool's help, then
    the tool's own verdict for the same clip is revealed for comparison.
    Fully separate from the live detection pipeline — operates on
    pre-loaded clips from config.PRACTICE_CLIPS_DIR, not live screen capture.
    """

    def __init__(self, shared_detector, shared_engine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Practice Mode")
        self.setModal(True)
        self.setFixedWidth(420)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #dddddd;
            }
            """
        )

        self._detector = shared_detector
        self._engine = shared_engine
        self._preprocessor = Preprocessor()

        self._clips = _list_practice_clips()
        random.shuffle(self._clips)
        self._clip_index = -1
        self._cap: cv2.VideoCapture | None = None
        self._user_guess = None
        self._correct_count = 0
        self._total_count = 0

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self._build_ui()

        if not self._clips:
            self._status_label.setText(
                f"No clips found under:\n{config.PRACTICE_CLIPS_DIR}\n\n"
                "Add .mp4/.avi/.mov/.mkv files to its real/ and deepfake/ subfolders."
            )
            self._real_btn.setEnabled(False)
            self._fake_btn.setEnabled(False)
        else:
            self._load_next_clip()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Practice Mode")
        title.setStyleSheet("font-size: 14px; font-weight: 500; color: #ffffff;")
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #333333;")
        layout.addWidget(line)

        self._video_widget = VideoWidget()
        self._video_widget.setFixedHeight(240)
        layout.addWidget(self._video_widget)

        self._status_label = QLabel("Watch the clip, then guess.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 12px; color: #aaaaaa;")
        layout.addWidget(self._status_label)

        guess_row = QHBoxLayout()
        guess_row.setSpacing(8)
        self._real_btn = QPushButton("Looks Real")
        self._real_btn.setFixedHeight(34)
        self._real_btn.clicked.connect(lambda: self._on_guess("REAL"))
        self._fake_btn = QPushButton("Looks Fake")
        self._fake_btn.setFixedHeight(34)
        self._fake_btn.clicked.connect(lambda: self._on_guess("DEEPFAKE"))
        guess_row.addWidget(self._real_btn)
        guess_row.addWidget(self._fake_btn)
        layout.addLayout(guess_row)

        self._score_label = QLabel("Score: 0/0")
        self._score_label.setStyleSheet("font-size: 11px; color: #888888;")
        layout.addWidget(self._score_label)

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("color: #333333;")
        layout.addWidget(line2)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._next_btn = QPushButton("Next Clip")
        self._next_btn.setFixedHeight(34)
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._load_next_clip)
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(34)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._next_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _load_next_clip(self):
        """Advance to the next clip (looping the shuffled list) and start blind playback."""
        self._clip_index = (self._clip_index + 1) % len(self._clips)
        path, _label = self._clips[self._clip_index]
        self._user_guess = None

        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        fps = self._cap.get(cv2.CAP_PROP_FPS) or config.DISPLAY_FPS
        self._play_timer.start(int(1000 / fps))

        self._status_label.setText("Watch the clip, then guess: is this face real or a deepfake?")
        self._real_btn.setEnabled(True)
        self._fake_btn.setEnabled(True)
        self._next_btn.setEnabled(False)

    def _advance_frame(self):
        """Push the next clip frame to the video widget, looping playback during the blind-guess phase."""
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                return
        self._video_widget.update_frame(frame, [], [])

    def _on_guess(self, guess: str):
        """Record the participant's blind guess, stop playback, and reveal the tool's verdict."""
        self._play_timer.stop()
        self._user_guess = guess
        self._real_btn.setEnabled(False)
        self._fake_btn.setEnabled(False)
        self._reveal()

    def _reveal(self):
        """Show the tool's verdict for the current clip and update the running participant score."""
        path, label = self._clips[self._clip_index]
        risk_level, p_fake = _classify_clip(path, self._detector, self._preprocessor, self._engine)

        self._total_count += 1
        if self._user_guess == label:
            self._correct_count += 1

        lines = [f"Your guess: {self._user_guess}", f"Actual label: {label}"]
        if risk_level is not None:
            lines.append(f"Tool verdict: {risk_level} ({p_fake * 100:.1f}%)")
            lines.append(
                "You and the tool agree." if self._user_guess == risk_level else "You and the tool disagree."
            )
        else:
            lines.append("Tool verdict: unavailable (no face detected / API error).")
        self._status_label.setText("\n".join(lines))
        self._score_label.setText(f"Score: {self._correct_count}/{self._total_count}")

        self._next_btn.setEnabled(True)

    def closeEvent(self, event):
        """Stop playback and release the video capture handle on close."""
        self._play_timer.stop()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)
