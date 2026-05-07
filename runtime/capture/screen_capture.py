"""
screen_capture.py

Captures the desktop screen using mss and returns BGR numpy frames.
"""

import numpy as np
import cv2
import mss
import mss.tools


class ScreenCapture:
    """
    Real-time screen capture using mss.

    Args:
        region (dict | None): Capture region with keys 'left', 'top', 'width', 'height'.
                              If None, captures the full primary monitor.
    """

    def __init__(self, region: dict = None):
        self._region = region
        self._sct = None

    def start(self):
        """Initialise the mss capture context."""
        self._sct = mss.mss()
        if self._region is None:
            monitor = self._sct.monitors[1]  # Primary monitor
            self._region = {
                "left": monitor["left"],
                "top": monitor["top"],
                "width": monitor["width"],
                "height": monitor["height"],
            }

    def read(self) -> np.ndarray:
        """
        Capture one frame from the screen.

        Returns:
            BGR numpy array of shape (height, width, 3).
        """
        if self._sct is None:
            raise RuntimeError("ScreenCapture not started. Call start() first.")

        screenshot = self._sct.grab(self._region)
        # mss returns BGRA; convert to BGR
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def set_region(self, left: int, top: int, width: int, height: int):
        """Update the capture region at runtime."""
        self._region = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

    def stop(self):
        """Release the mss context."""
        if self._sct:
            self._sct.close()
            self._sct = None
