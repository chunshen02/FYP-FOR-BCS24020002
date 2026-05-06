#!/usr/bin/env python3
"""Quick test of history table functionality."""

import sys
from PyQt6.QtWidgets import QApplication
from runtime.gui.main_window import MainWindow

# Create application
app = QApplication(sys.argv)
window = MainWindow()

# Check that history section is properly configured
print("✓ History section visible:", window._history_section.isVisible())
print("✓ History table row count:", window._history_table.rowCount())
print("✓ History table column count:", window._history_table.columnCount())

# Get column headers
headers = [window._history_table.horizontalHeaderItem(i).text() for i in range(4)]
print("✓ Column headers:", headers)
assert headers == ["Date & time", "Duration", "Alerts", "Result"]

# Test populate with sample data
sessions_test = [
    {
        "started_at": "2026-05-06 10:00:00",
        "duration": "00:05:30",
        "alert_count": 2,
        "result": "Deepfake detected"
    },
    {
        "started_at": "2026-05-06 09:00:00",
        "duration": "00:03:15",
        "alert_count": 0,
        "result": "Clean"
    }
]

window.populate_history_table(sessions_test)
print("✓ After populate - row count:", window._history_table.rowCount())
assert window._history_table.rowCount() == 2

# Verify table content
dt_item = window._history_table.item(0, 0)
print("✓ First row date&time:", dt_item.text())
assert dt_item.text() == "2026-05-06 10:00:00"

dur_item = window._history_table.item(0, 1)
print("✓ First row duration:", dur_item.text())
assert dur_item.text() == "00:05:30"

alert_item = window._history_table.item(0, 2)
print("✓ First row alerts:", alert_item.text())
assert alert_item.text() == "2"

# Test refresh_history (should load from database)
print("\nTesting refresh_history()...")
window.refresh_history()
print("✓ Refresh complete - row count:", window._history_table.rowCount())
print("✓ History section visible:", window._history_section.isVisible())

print("\n✅ All tests passed!")
