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
import numpy as np
from dataclasses import dataclass, field
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


class BitmindAPIError(Exception):
    """Base class for all BitMind API failures."""


class BitmindAuthError(BitmindAPIError):
    """Invalid or revoked API key (HTTP 401/403) — never retried."""


class BitmindRateLimitError(BitmindAPIError):
    """Rate limit exceeded (HTTP 429) — distinct from a transient failure, not retried inline."""


class BitmindTransientError(BitmindAPIError):
    """Timeout, connection error, or server error (5xx) — retried with backoff."""


@dataclass
class FaceResult:
    """Holds inference results for a single detected face."""
    p_fake_raw: float        # Raw P_fake from API — used for alert triggering and display
    risk_level: str          # "REAL" or "DEEPFAKE"
    api_response_ms: float   # Round-trip API response time in milliseconds
    bbox: dict               # {"x", "y", "w", "h"} in frame pixels
    last_updated: float = 0.0  # time.perf_counter() at last API call
    explanation: list = field(default_factory=list)  # local CV evidence strings, populated only for DEEPFAKE
    thumbnail: np.ndarray = None  # the exact preprocessed BGR crop that was analyzed, for Mini mode display
    face_id: str = ""        # tracker-assigned id — lets the GUI target a manual re-check at the right face


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
        """
        Submit one face crop to the BitMind API, retrying transient failures
        internally until success or final give-up. Raises BitmindAPIError on
        failure. Used by offline batch evaluation, which processes samples
        sequentially with no round-robin queue to block.
        """
        p_fake_raw, api_ms = self._call_api(jpeg_bytes)
        return self._to_result(p_fake_raw, api_ms, bbox, face_id)

    def run_single_attempt(self, jpeg_bytes: bytes, bbox: dict, face_id: str) -> "FaceResult":
        """
        Submit one face crop with exactly one HTTP attempt — no internal
        retry loop. Used by the live round-robin pipeline so a failing face
        yields its turn instead of blocking the queue through a multi-attempt
        backoff; the caller re-queues on failure. Raises BitmindAPIError on
        failure.
        """
        p_fake_raw, api_ms = self._post_once(jpeg_bytes)
        return self._to_result(p_fake_raw, api_ms, bbox, face_id)

    def _to_result(self, p_fake_raw: float, api_ms: float, bbox: dict, face_id: str) -> "FaceResult":
        return FaceResult(
            p_fake_raw=p_fake_raw,
            risk_level=_get_risk_level(p_fake_raw),
            api_response_ms=api_ms,
            bbox=bbox,
            last_updated=time.perf_counter(),
            face_id=face_id,
        )

    def _throttle(self):
        """Sleep as needed to enforce the 1 request/second rate limit on the free tier."""
        now = time.perf_counter()
        elapsed = now - self._last_call_time
        if elapsed < config.BITMIND_RATE_LIMIT_DELAY_SEC:
            time.sleep(config.BITMIND_RATE_LIMIT_DELAY_SEC - elapsed)

    def _post_once(self, jpeg_bytes: bytes) -> tuple:
        """
        Perform a single HTTP attempt against the BitMind API.

        Raises BitmindAuthError/BitmindRateLimitError immediately (not
        retryable); raises BitmindTransientError for timeout, connection
        error, or 5xx (retryable by the caller); raises BitmindAPIError
        for any other failure.

        Returns:
            (p_fake_raw: float, api_response_ms: float)
        """
        self._throttle()
        t_start = time.perf_counter()
        try:
            b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            response = self._session.post(
                config.BITMIND_API_ENDPOINT,
                json={"image": f"data:image/jpeg;base64,{b64}"},
                timeout=config.BITMIND_API_TIMEOUT_SEC
            )
        except requests.exceptions.Timeout as e:
            self._last_call_time = time.perf_counter()
            raise BitmindTransientError(f"Request timed out after {config.BITMIND_API_TIMEOUT_SEC}s") from e
        except requests.exceptions.ConnectionError as e:
            # Covers both DNS failure and connection refused — requests wraps both here.
            self._last_call_time = time.perf_counter()
            raise BitmindTransientError(f"Connection error: {e}") from e
        except requests.exceptions.RequestException as e:
            self._last_call_time = time.perf_counter()
            raise BitmindAPIError(f"Request failed: {e}") from e

        t_end = time.perf_counter()
        api_ms = (t_end - t_start) * 1000
        self._last_call_time = t_end

        if response.status_code == 200:
            data = response.json()
            is_ai = data.get("isAI", False)
            confidence = float(data.get("confidence", 0.0))
            p_fake = confidence if is_ai else 1.0 - confidence
            return p_fake, api_ms
        if response.status_code in (401, 403):
            raise BitmindAuthError(f"Authentication failed (HTTP {response.status_code}): {response.text}")
        if response.status_code == 429:
            raise BitmindRateLimitError(f"Rate limit exceeded (HTTP 429): {response.text}")
        if response.status_code >= 500:
            raise BitmindTransientError(f"Server error (HTTP {response.status_code}): {response.text}")
        raise BitmindAPIError(f"Unexpected HTTP {response.status_code}: {response.text}")

    def _call_api(self, jpeg_bytes: bytes) -> tuple:
        """
        Submit one JPEG image to the BitMind API, retrying only transient
        failures (timeout, connection error, 5xx) with exponential backoff.
        Auth (401/403) and rate-limit (429) failures propagate immediately.

        Returns:
            (p_fake_raw: float, api_response_ms: float)
        """
        delays = [0.0] + config.BITMIND_RETRY_DELAYS_SEC
        last_error = None
        for delay in delays:
            if delay:
                time.sleep(delay)
            try:
                return self._post_once(jpeg_bytes)
            except BitmindTransientError as e:
                last_error = e
        raise last_error

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()
