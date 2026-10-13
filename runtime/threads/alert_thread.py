"""
alert_thread.py

Lightweight QThread that monitors FaceResult streams and manages
alert trigger and dismiss logic.
"""

import queue
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class AlertThread(QThread):
    """
    Monitors incoming FaceResult lists and raises one alert per deepfake face.

    Signals:
        alert_triggered(list, int): Emitted on a frame where faces newly become
            alerts. Carries each new face's P_fake (highest first) and how many
            DEEPFAKE faces are on screen in that frame.
        alert_dismissed():          Emitted once no face has been DEEPFAKE for
            ALERT_DISMISS_FRAMES frames.
    """

    alert_triggered = pyqtSignal(list, int)
    alert_dismissed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue(maxsize=60)
        self._running = False
        self._alert_active = False
        self._red_streaks: dict = {}   # face_id -> consecutive frames it has been DEEPFAKE
        # Every face that has alerted this session. Track ids are never reused
        # within a session, so a face alerts once however long it stays on
        # screen, and a brief detection dropout does not re-alert it.
        self._alerted_ids: set = set()
        self._dismiss_counter = 0

    def push_results(self, results: list):
        """
        Push a list of FaceResult objects from the detection worker.
        Called from the GUI thread via signal connection.
        Non-blocking: drops frames if queue is full.
        """
        try:
            self._queue.put_nowait(results)
        except queue.Full:
            pass

    def run(self):
        self._running = True

        while self._running:
            try:
                results: list = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            fakes = [r for r in results if r.risk_level == "DEEPFAKE"]
            self._red_streaks = {r.face_id: self._red_streaks.get(r.face_id, 0) + 1 for r in fakes}

            # Each face is debounced on its own, so a second deepfake joining
            # an alert already on screen (or one the user acknowledged) still
            # raises its own alert instead of being absorbed by the first.
            new_alerts = [
                r for r in fakes
                if r.face_id not in self._alerted_ids
                and self._red_streaks[r.face_id] >= config.ALERT_TRIGGER_FRAMES
            ]
            if new_alerts:
                self._alerted_ids.update(r.face_id for r in new_alerts)
                self._alert_active = True
                self._dismiss_counter = 0
                peaks = sorted((r.p_fake_raw for r in new_alerts), reverse=True)
                self.alert_triggered.emit(peaks, len(fakes))
            elif self._alert_active:
                if fakes:
                    self._dismiss_counter = 0
                else:
                    self._dismiss_counter += 1
                    if self._dismiss_counter >= config.ALERT_DISMISS_FRAMES:
                        self._alert_active = False
                        self._dismiss_counter = 0
                        self.alert_dismissed.emit()

    def stop(self):
        self._running = False
        self._alert_active = False
        self._red_streaks = {}
        self._alerted_ids = set()
        self._dismiss_counter = 0
