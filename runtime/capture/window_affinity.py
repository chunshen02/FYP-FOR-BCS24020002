"""
window_affinity.py

Excludes this app's own window from screen capture at the OS compositor
level via the Windows SetWindowDisplayAffinity API, so the app's window
never reaches any capture buffer (ours or anyone else's) in the first
place. This is the primary self-exclusion mechanism; DetectionWorker's
pixel-masking (_mask_exclusion) remains as a defensive fallback for
non-Windows platforms or if this call fails.

capture_affinity_lock guards the brief window where exclusion is
temporarily cleared (e.g. for a manual full-screen screenshot that should
include the app's own window) so DetectionWorker's continuous capture loop
can never grab a frame while the app is momentarily un-excluded.
"""

import platform
import sys
import threading

WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Windows 10 2004+ (build 19041+) only

capture_affinity_lock = threading.Lock()


def _windows_build() -> int:
    """Return the Windows build number, or 0 if it can't be determined."""
    try:
        return sys.getwindowsversion().build
    except AttributeError:
        return 0


def _set_affinity(hwnd: int, affinity: int) -> bool:
    """Low-level SetWindowDisplayAffinity call. False on non-Windows or any failure."""
    if platform.system() != "Windows":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
        return bool(user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), wintypes.DWORD(affinity)))
    except Exception:
        return False


def apply_capture_exclusion(hwnd: int) -> tuple:
    """
    Exclude the given native window handle from screen capture.

    Uses WDA_EXCLUDEFROMCAPTURE on Windows 10 2004+ (fully excluded from
    any capture, including GDI BitBlt-based capture such as mss uses), or
    falls back to the older WDA_MONITOR flag on earlier Windows 10/11
    builds. No-ops on non-Windows platforms.

    Returns (description, applied_affinity): a short human-readable
    description for logging, and the WDA_* constant that actually took
    effect (None if exclusion isn't active — non-Windows, old build, or
    the call failed). Callers that need to temporarily lift exclusion
    (see clear_capture_exclusion/restore_capture_exclusion) must hang on
    to applied_affinity to restore the correct value afterward.
    """
    if platform.system() != "Windows":
        return "pixel masking (fallback — non-Windows platform)", None

    build = _windows_build()
    affinity = WDA_EXCLUDEFROMCAPTURE if build >= 19041 else WDA_MONITOR

    if not _set_affinity(hwnd, affinity):
        return "pixel masking (fallback — SetWindowDisplayAffinity call failed)", None

    if affinity == WDA_EXCLUDEFROMCAPTURE:
        return "SetWindowDisplayAffinity (native, WDA_EXCLUDEFROMCAPTURE)", affinity
    return f"SetWindowDisplayAffinity (native, WDA_MONITOR fallback — Windows build {build} < 19041)", affinity


def clear_capture_exclusion(hwnd: int) -> bool:
    """Temporarily make hwnd capturable again (WDA_NONE). Pair with restore_capture_exclusion."""
    return _set_affinity(hwnd, WDA_NONE)


def restore_capture_exclusion(hwnd: int, affinity: int) -> bool:
    """Restore a previously-active exclusion affinity value (from apply_capture_exclusion)."""
    return _set_affinity(hwnd, affinity)
