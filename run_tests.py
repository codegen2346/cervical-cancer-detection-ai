"""
Convenience test runner for the Cervical Cancer Detection Research MVP.
Runs pytest with verbose reporting and summary statistics.
"""

import sys
import pytest

if __name__ == "__main__":
    print("=" * 60)
    print("Running Cervical Cancer AI Detection Test Suite...")
    print("=" * 60)
    exit_code = pytest.main(["tests/", "-v", "--tb=short"])
    if exit_code == 0:
        print("\nAll unit tests passed successfully!")
    else:
        print(f"\nTests failed with exit code: {exit_code}")
    sys.exit(exit_code)
