"""
global_hotkey.py

System-wide keyboard shortcut via the Windows RegisterHotKey API, so it
fires while the meeting or browser window has focus rather than only when
this app's own window does.
"""

import ctypes
import platform
from ctypes import wintypes

from PyQt6.QtCore import QAbstractNativeEventFilter, Qt
from PyQt6.QtGui import QKeySequence

_WM_HOTKEY = 0x0312
_HOTKEY_ID = 1
_MOD_ALT, _MOD_CONTROL, _MOD_SHIFT, _MOD_WIN, _MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
_VK_F1 = 0x70
_MODIFIER_FLAGS = (
    (Qt.KeyboardModifier.ControlModifier, _MOD_CONTROL),
    (Qt.KeyboardModifier.AltModifier, _MOD_ALT),
    (Qt.KeyboardModifier.ShiftModifier, _MOD_SHIFT),
    (Qt.KeyboardModifier.MetaModifier, _MOD_WIN),
)


def hotkey_parts(sequence: str) -> tuple | None:
    """Convert a portable key sequence such as "Ctrl+Alt+R" to (MOD_* flags, virtual-key code), or None if unsupported."""
    keys = QKeySequence(sequence, QKeySequence.SequenceFormat.PortableText)
    if keys.count() != 1:
        return None
    combination = keys[0]
    key = combination.key().value
    modifiers = combination.keyboardModifiers()
    flags = 0
    for qt_modifier, flag in _MODIFIER_FLAGS:
        if modifiers & qt_modifier:
            flags |= flag

    if Qt.Key.Key_F1.value <= key <= Qt.Key.Key_F24.value:
        return flags, _VK_F1 + key - Qt.Key.Key_F1.value
    # A letter or digit with only Shift (or nothing) held would swallow ordinary typing in every app.
    if not flags & (_MOD_CONTROL | _MOD_ALT | _MOD_WIN):
        return None
    # Qt's key codes for A-Z and 0-9 are their ASCII codes, which are also their virtual-key codes.
    if Qt.Key.Key_A.value <= key <= Qt.Key.Key_Z.value or Qt.Key.Key_0.value <= key <= Qt.Key.Key_9.value:
        return flags, key
    return None


class GlobalHotkey(QAbstractNativeEventFilter):
    """Calls callback whenever the registered system-wide shortcut is pressed; install with installNativeEventFilter."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._registered = False

    def register(self, sequence: str) -> bool:
        """Replace any current shortcut with sequence; False if it is unsupported or another app already holds it."""
        self.unregister()
        parts = hotkey_parts(sequence)
        if parts is None or platform.system() != "Windows":
            return False
        modifiers, vk = parts
        # No window handle: WM_HOTKEY then goes to this (GUI) thread's queue, which Qt's
        # event dispatcher passes to nativeEventFilter.
        self._registered = bool(ctypes.windll.user32.RegisterHotKey(None, _HOTKEY_ID, modifiers | _MOD_NOREPEAT, vk))
        return self._registered

    def unregister(self):
        if self._registered:
            ctypes.windll.user32.UnregisterHotKey(None, _HOTKEY_ID)
            self._registered = False

    def nativeEventFilter(self, event_type, message):
        # The result must be 0, not None: Qt writes it through to a window procedure's LRESULT,
        # and None crashes the app (access violation) as soon as a window is shown.
        if self._registered and event_type == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == _WM_HOTKEY and msg.wParam == _HOTKEY_ID:
                self._callback()
                return True, 0
        return False, 0
