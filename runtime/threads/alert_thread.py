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
from runtime.detection.inference import FaceResult


class AlertThread(QThread):
    """
    Monitors incoming FaceResult lists and manages alert state.

    Signals:
        alert_triggered(float): Emitted when alert fires. Carries peak P_fake.
        alert_dismissed():      Emitted when alert is dismissed.
    """

    alert_triggered = pyqtSignal(float)
    alert_dismissed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue(maxsize=60)
        self._running = False
        self._alert_active = False
        self._trigger_counter = 0
        self._dismiss_counter = 0

    def push_results(self, results: list):
        """
        Push a list of FaceResult objects from the detection worker.
        Called from the GUI thread via signal connection.
        Non-blocking: drops frames if queue is full.
        """
        _ = FaceResult  # keeps explicit type intent without runtime dependency use
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

            # Determine if any face in this frame is RED
            is_red_frame = any(r.risk_level == "RED" for r in results)
            peak_p_fake = max((r.p_fake_raw for r in results), default=0.0)

            if not self._alert_active:
                if is_red_frame:
                    self._trigger_counter += 1
                    self._dismiss_counter = 0
                    if self._trigger_counter >= config.ALERT_TRIGGER_FRAMES:
                        self._alert_active = True
                        self._trigger_counter = 0
                        self.alert_triggered.emit(peak_p_fake)
                else:
                    self._trigger_counter = 0
            else:
                if not is_red_frame:
                    self._dismiss_counter += 1
                    self._trigger_counter = 0
                    if self._dismiss_counter >= config.ALERT_DISMISS_FRAMES:
                        self._alert_active = False
                        self._dismiss_counter = 0
                        self.alert_dismissed.emit()
                else:
                    self._dismiss_counter = 0

    def stop(self):
        self._running = False
        self._alert_active = False
        self._trigger_counter = 0
        self._dismiss_counter = 0
