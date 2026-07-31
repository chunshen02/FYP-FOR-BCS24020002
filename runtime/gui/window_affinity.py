"""
window_affinity.py

Excludes this app's own window from screen capture at the OS compositor
level via the Windows SetWindowDisplayAffinity API, so the app's window
never reaches any capture buffer (ours or anyone else's) in the first
place. This is the primary self-exclusion mechanism; DetectionWorker's
pixel-masking (_mask_exclusion) remains as a defensive fallback for
non-Windows platforms or if this call fails.
"""

import platform
import sys

WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Windows 10 2004+ (build 19041+) only


def _windows_build() -> int:
    """Return the Windows build number, or 0 if it can't be determined."""
    try:
        return sys.getwindowsversion().build
    except AttributeError:
        return 0


def apply_capture_exclusion(hwnd: int) -> str:
    """
    Exclude the given native window handle from screen capture.

    Uses WDA_EXCLUDEFROMCAPTURE on Windows 10 2004+ (fully excluded from
    any capture, including GDI BitBlt-based capture such as mss uses), or
    falls back to the older WDA_MONITOR flag on earlier Windows 10/11
    builds. No-ops on non-Windows platforms.

    Returns a short human-readable description of the active mechanism,
    for logging.
    """
    if platform.system() != "Windows":
        return "pixel masking (fallback — non-Windows platform)"

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.SetWindowDisplayAffinity.restype = wintypes.BOOL

        build = _windows_build()
        affinity = WDA_EXCLUDEFROMCAPTURE if build >= 19041 else WDA_MONITOR

        ok = user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), wintypes.DWORD(affinity))
        if not ok:
            return "pixel masking (fallback — SetWindowDisplayAffinity call failed)"

        if affinity == WDA_EXCLUDEFROMCAPTURE:
            return "SetWindowDisplayAffinity (native, WDA_EXCLUDEFROMCAPTURE)"
        return f"SetWindowDisplayAffinity (native, WDA_MONITOR fallback — Windows build {build} < 19041)"

    except Exception as e:
        return f"pixel masking (fallback — SetWindowDisplayAffinity unavailable: {e})"
