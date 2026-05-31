"""
inference.py

Submits JPEG face crop bytes to the BitMind API and returns
the raw fake confidence score.

BitMind API:
    Endpoint : https://api.bitmind.ai/detect-image
    Method   : HTTP POST
    Auth     : Bearer token (API key from .env)
    Body     : JSON { "image": "data:image/jpeg;base64,<b64>" }
    Response : JSON { "isAI": <bool>, "confidence": <float 0.0-1.0>, ... }
"""

import base64
import os
import sys
import time
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


@dataclass
class FaceResult:
    """Holds inference results for a single detected face."""
    p_fake_raw: float        # Raw P_fake from API — used for alert triggering and display
    risk_level: str          # "REAL" or "DEEPFAKE"
    api_response_ms: float   # Round-trip API response time in milliseconds
    bbox: dict               # {"x", "y", "w", "h"} in frame pixels
    last_updated: float = 0.0  # time.perf_counter() at last API call


def _get_risk_level(p_fake: float) -> str:
    return "DEEPFAKE" if p_fake >= config.FAKE_THRESHOLD else "REAL"


class DeepfakeInferenceEngine:
    """Submits face crop images to the BitMind API for deepfake classification."""

    def __init__(self):
        self._api_key = os.environ.get("BITMIND_API_KEY", "")
        if not self._api_key:
            raise EnvironmentError(
                "BITMIND_API_KEY not found. "
                "Add it to the .env file at the project root: BITMIND_API_KEY=your_key_here"
            )

        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})
        self._last_call_time: float = 0.0

    def run_single(self, jpeg_bytes: bytes, bbox: dict, face_id: str) -> "FaceResult":
        """Submit one face crop to the BitMind API and return its result."""
        p_fake_raw, api_ms = self._call_api(jpeg_bytes)
        return FaceResult(
            p_fake_raw=p_fake_raw,
            risk_level=_get_risk_level(p_fake_raw),
            api_response_ms=api_ms,
            bbox=bbox,
            last_updated=time.perf_counter(),
        )

    def _call_api(self, jpeg_bytes: bytes) -> tuple:
        """
        Submit one JPEG image to the BitMind API.

        Enforces the 1 request/second rate limit on the free tier.

        Returns:
            (p_fake_raw: float, api_response_ms: float)
        """
        now = time.perf_counter()
        elapsed = now - self._last_call_time
        if elapsed < config.BITMIND_RATE_LIMIT_DELAY_SEC:
            time.sleep(config.BITMIND_RATE_LIMIT_DELAY_SEC - elapsed)

        t_start = time.perf_counter()
        try:
            b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            response = self._session.post(
                config.BITMIND_API_ENDPOINT,
                json={"image": f"data:image/jpeg;base64,{b64}"},
                timeout=config.BITMIND_API_TIMEOUT_SEC
            )
            api_ms = (time.perf_counter() - t_start) * 1000
            self._last_call_time = time.perf_counter()

            if response.status_code == 200:
                data = response.json()
                is_ai = data.get("isAI", False)
                confidence = float(data.get("confidence", 0.0))
                p_fake = confidence if is_ai else 1.0 - confidence
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

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()
