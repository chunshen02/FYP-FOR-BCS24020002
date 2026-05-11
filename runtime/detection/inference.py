"""
inference.py

Submits JPEG face crop bytes to the BitMind API and returns
raw and EMA-smoothed fake confidence scores.

BitMind API:
    Endpoint : https://api.bitmind.ai/oracle/v1/image-detect
    Method   : HTTP POST
    Auth     : Bearer token (API key from .env)
    Body     : multipart/form-data with field 'image' containing JPEG bytes
    Response : JSON { "result": { "fake": <float 0.0-1.0>, ... } }
"""

import os
import sys
import time
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

# Load API key from .env file at project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


@dataclass
class FaceResult:
    """Holds inference results for a single detected face."""
    p_fake_raw: float        # Raw P_fake from API — used for alert triggering
    p_fake_display: float    # EMA-smoothed P_fake — used for visual overlay
    risk_level: str          # "GREEN", "AMBER", or "RED"
    api_response_ms: float   # Round-trip API response time in milliseconds


def _get_risk_level(p_fake: float) -> str:
    if p_fake >= config.FAKE_THRESHOLD:
        return "RED"
    elif p_fake >= config.WARNING_THRESHOLD:
        return "AMBER"
    else:
        return "GREEN"


class DeepfakeInferenceEngine:
    """
    Submits face crop images to the BitMind API for deepfake classification.
    Maintains per-face EMA state for smoothed display scores.
    """

    def __init__(self):
        self._api_key = os.environ.get("BITMIND_API_KEY", "")
        if not self._api_key:
            raise EnvironmentError(
                "BITMIND_API_KEY not found. "
                "Add it to the .env file at the project root: BITMIND_API_KEY=your_key_here"
            )

        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})
        self._ema_state: list = []   # One smoothed score per face slot
        self._last_call_time: float = 0.0

    def run(self, jpeg_bytes_list: list) -> list:
        """
        Submit a list of JPEG face crop byte payloads to the BitMind API.

        Args:
            jpeg_bytes_list: List of bytes objects, one per detected face.
                             Each element is the JPEG output of Preprocessor.process().

        Returns:
            List of FaceResult objects, one per face.
            Returns an empty list if jpeg_bytes_list is empty.
        """
        results = []

        # Extend EMA state if more faces appear than previously tracked
        while len(self._ema_state) < len(jpeg_bytes_list):
            self._ema_state.append(0.0)

        for i, jpeg_bytes in enumerate(jpeg_bytes_list):
            p_fake_raw, api_ms = self._call_api(jpeg_bytes)

            # EMA smoothing for display
            prev_ema = self._ema_state[i]
            p_fake_display = (config.EMA_ALPHA * p_fake_raw
                              + (1 - config.EMA_ALPHA) * prev_ema)
            self._ema_state[i] = p_fake_display

            results.append(FaceResult(
                p_fake_raw=p_fake_raw,
                p_fake_display=p_fake_display,
                risk_level=_get_risk_level(p_fake_raw),  # use raw for risk
                api_response_ms=api_ms
            ))

        # Trim EMA state if fewer faces appear this frame
        if len(jpeg_bytes_list) < len(self._ema_state):
            self._ema_state = self._ema_state[:len(jpeg_bytes_list)]

        return results

    def _call_api(self, jpeg_bytes: bytes) -> tuple:
        """
        Submit one JPEG image to the BitMind API.

        Enforces the 1 request/second rate limit on the free tier.

        Returns:
            (p_fake_raw: float, api_response_ms: float)
        """
        # Rate limiting: enforce minimum gap between calls
        now = time.perf_counter()
        elapsed = now - self._last_call_time
        if elapsed < config.BITMIND_RATE_LIMIT_DELAY_SEC:
            time.sleep(config.BITMIND_RATE_LIMIT_DELAY_SEC - elapsed)

        t_start = time.perf_counter()
        try:
            response = self._session.post(
                config.BITMIND_API_ENDPOINT,
                files={"image": ("face.jpg", jpeg_bytes, "image/jpeg")},
                timeout=config.BITMIND_API_TIMEOUT_SEC
            )
            api_ms = (time.perf_counter() - t_start) * 1000
            self._last_call_time = time.perf_counter()

            if response.status_code == 200:
                data = response.json()
                # BitMind response format: { "result": { "fake": <float> } }
                p_fake = float(data.get("result", {}).get("fake", 0.0))
                return p_fake, api_ms
            else:
                print(f"[WARNING] BitMind API returned status {response.status_code}: {response.text}")
                return 0.0, api_ms

        except requests.exceptions.Timeout:
            api_ms = (time.perf_counter() - t_start) * 1000
            self._last_call_time = time.perf_counter()
            print(f"[WARNING] BitMind API request timed out after {config.BITMIND_API_TIMEOUT_SEC}s")
            return 0.0, api_ms

        except requests.exceptions.RequestException as e:
            api_ms = (time.perf_counter() - t_start) * 1000
            self._last_call_time = time.perf_counter()
            print(f"[WARNING] BitMind API request failed: {e}")
            return 0.0, api_ms

    def reset_ema(self):
        """Reset EMA state. Call when detection is stopped and restarted."""
        self._ema_state = []

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()
