#!/usr/bin/env python3
"""Debug test for get_all_sessions."""

from runtime.storage.database import get_all_sessions

# Test the database function directly
print("Testing get_all_sessions()...")
sessions = get_all_sessions()
print(f"Sessions returned: {len(sessions)}")
print(f"Sessions: {sessions}")

# Test the logic
if not sessions:
    print("-> Would hide section (no sessions)")
else:
    print(f"-> Would show section and populate with {len(sessions)} session(s)")
