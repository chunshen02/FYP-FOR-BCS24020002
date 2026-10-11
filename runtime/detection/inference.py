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
    """Rate limit exceeded (HTTP 429); retry_after is the server's requested pause in seconds, if given."""

    def __init__(self, message: str, retry_after: float | None):
        super().__init__(message)
        self.retry_after = retry_after


class BitmindTransientError(BitmindAPIError):
    """Timeout, connection error, or server error (5xx)."""


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


def risk_level_for(p_fake: float) -> str:
    """Map a P_fake score to "DEEPFAKE" or "REAL" using the current FAKE_THRESHOLD."""
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

    def run_single_attempt(self, jpeg_bytes: bytes, bbox: dict, face_id: str) -> "FaceResult":
        """Submit one face crop with a single HTTP attempt; raises BitmindAPIError on failure."""
        p_fake_raw, api_ms = self._post_once(jpeg_bytes)
        return FaceResult(
            p_fake_raw=p_fake_raw,
            risk_level=risk_level_for(p_fake_raw),
            api_response_ms=api_ms,
            bbox=bbox,
            last_updated=time.perf_counter(),
            face_id=face_id,
        )

    def _throttle(self):
        """Space request starts at least BITMIND_RATE_LIMIT_DELAY_SEC apart (1 request/second free tier)."""
        # Timed from the previous request's start, not its end: with ~1.8 s responses, waiting a further
        # second after each response held throughput to one call per ~2.8 s for no rate-limit benefit.
        elapsed = time.perf_counter() - self._last_call_time
        if elapsed < config.BITMIND_RATE_LIMIT_DELAY_SEC:
            time.sleep(config.BITMIND_RATE_LIMIT_DELAY_SEC - elapsed)
        self._last_call_time = time.perf_counter()

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
            raise BitmindTransientError(f"Request timed out after {config.BITMIND_API_TIMEOUT_SEC}s") from e
        except requests.exceptions.ConnectionError as e:
            # Covers both DNS failure and connection refused — requests wraps both here.
            raise BitmindTransientError(f"Connection error: {e}") from e
        except requests.exceptions.RequestException as e:
            raise BitmindAPIError(f"Request failed: {e}") from e

        api_ms = (time.perf_counter() - t_start) * 1000

        if response.status_code == 200:
            # Defaulting missing fields would record a P_fake BitMind never returned (isAI=False,
            # confidence=0 reads as 100% fake), so a malformed body is a failure, not a score.
            try:
                data = response.json()
                is_ai = data["isAI"]
                confidence = float(data["confidence"])
            except (ValueError, KeyError, TypeError) as e:
                raise BitmindAPIError(f"Malformed response: {response.text}") from e
            p_fake = confidence if is_ai else 1.0 - confidence
            return p_fake, api_ms
        if response.status_code in (401, 403):
            raise BitmindAuthError(f"Authentication failed (HTTP {response.status_code}): {response.text}")
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "")
            raise BitmindRateLimitError(f"Rate limit exceeded (HTTP 429): {response.text}",
                                        float(retry_after) if retry_after.isdigit() else None)
        if response.status_code >= 500:
            raise BitmindTransientError(f"Server error (HTTP {response.status_code}): {response.text}")
        raise BitmindAPIError(f"Unexpected HTTP {response.status_code}: {response.text}")

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()
